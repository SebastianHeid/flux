import copy
from typing import List, Literal, Optional, Tuple

import library.train_util as train_util
import torch
import torch.nn as nn
from GRASP.model_parts import DoubleStreamBlockGRASP, GRASPLayer, SingleStreamBlockGRASP, SVDLinear
from library import (
    deepspeed_utils,
    flux_train_utils,
    flux_utils,
    strategy_base,
    strategy_flux,
)
from library.custom_train_functions import apply_masked_loss
from torch.utils.data import DataLoader
from tqdm import tqdm


def adaptive_rank_selection(svd_importance_list, target_ratio):
    total_sum = sum(svd_importance_list)
    target_sum = total_sum * target_ratio

    sorted_list = sorted(enumerate(svd_importance_list), key=lambda x: -x[1])

    cumulative_sum = 0
    indices = []
    for index, value in sorted_list:
        cumulative_sum += value
        indices.append(index)
        if cumulative_sum >= target_sum:
            break
    return indices

class GRASPBaseModel(nn.Module):
    def __init__(self, model: nn.Module, 
                 accelerator, 
                 text_encoding_strategy,  
                 flux_tokenize_strategy,
                 clip_l,
                 t5xxl,
                 ae,
                 weight_dtype,
                 noise_scheduler,
                 args) -> None:
        super(GRASPBaseModel, self).__init__()
        self.model = model
        self.accelerator = accelerator
        self.args=args
        self.text_encoding_strategy = text_encoding_strategy
        self.flux_tokenize_strategy = flux_tokenize_strategy
        self.clip_l = clip_l
        self.t5xxl=t5xxl
        self.weight_dtype = weight_dtype
        self.noise_scheduler = noise_scheduler
        self.ae = ae
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        self.grasp_values_dict = {}
        
        
    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_model = model
        for token in tokens[:-1]:
            sub_model = getattr(sub_model, token)
        setattr(sub_model, tokens[-1], module)
        
    def replace_double_blocks_with_GRASPLayer(self, target_layer: int,  device: Literal["cuda", "cpu"] = "cuda", log_file: Optional[str] = None):
        self.model.double_blocks[target_layer] = DoubleStreamBlockGRASP( self.model.double_blocks[target_layer],
                                                                        rank_attn_img=self.args.double_rank_img_attn,
                                                                        rank_attn_txt=self.args.double_rank_txt_attn,
                                                                        rank_img_mlp_in=self.args.double_rank_img_mlp,
                                                                        rank_img_mlp_out=self.args.double_rank_img_mlp,
                                                                        rank_txt_mlp_in=self.args.double_rank_txt_mlp,
                                                                        rank_txt_mlp_out=self.args.double_rank_txt_mlp,
                                                                        rank_img_mod=self.args.double_rank_img_mod,
                                                                        rank_txt_mod=self.args.double_rank_txt_mod,
                                                                        flag_img_attn=self.args.double_flag_img_attn,
                                                                        flag_txt_attn=self.args.double_flag_txt_attn,
                                                                        flag_img_mlp=self.args.double_flag_img_mlp,
                                                                        flag_txt_mlp=self.args.double_flag_txt_mlp,
                                                                        flag_img_mod=self.args.double_flag_img_mod,
                                                                        flag_txt_mod=self.args.double_flag_txt_mod)
    
    def replace_single_blocks_with_GRASPLayer(self, target_layer: int,  device: Literal["cuda", "cpu"] = "cuda", log_file: Optional[str] = None):
        self.model.single_blocks[target_layer] = SingleStreamBlockGRASP(self.model.single_blocks[target_layer],
                                                                        rank_attn=self.args.single_rank_attn,
                                                                        rank_mlp=self.args.single_rank_mlp,
                                                                        rank_mlp2=self.args.single_rank_mlp2,
                                                                        rank_mod=self.args.single_rank_mod,
                                                                        flag_mod=self.args.single_flag_mod,
                                                                        flag_attn=self.args.single_flag_attn,
                                                                        flag_mlp=self.args.single_flag_mlp,
                                                                        flag_mlp2=self.args.single_flag_mlp2)
        
    def compress_double_block(
        self,
        layer_id: int,
        ):
        self.replace_double_blocks_with_GRASPLayer(target_layer=layer_id)       
    
    def compress_single_block(
        self,
        layer_id: int,
        ):
        self.replace_single_blocks_with_GRASPLayer(target_layer=layer_id) 
        
    def check_exists_grasp_layer(self, log_file: Optional[str] = None):
        grasp_layer_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, GRASPLayer):
                grasp_layer_names.append(name)
                continue
        if not grasp_layer_names:
            print("GRASPLayer not found in current model, please use GRASPBaseModel.replace_with_GRASPLayer first")
        return grasp_layer_names
    
    
    from typing import Tuple

    import torch

    def compute_preserve_rank(self, grasp_layer: GRASPLayer, compression_ratio: float):
        if compression_ratio is None:
            raise ValueError("Compression ratio should not be None")
        in_features = grasp_layer.in_features
        out_features = grasp_layer.out_features
        k = int(in_features * out_features * (1 - compression_ratio) / (in_features + out_features))
        return k
    
    def get_noisy_model_input_at_timestep(
        self,
        target_timestep: int,  # NEU: Ein Integer, z.B. 0, 100, 500, 999 
        latents: torch.Tensor, 
        noise: torch.Tensor, 
        device, 
        dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Erzeugt ein verrauschtes 'noisy_model_input' für einen *spezifischen* vorgegebenen Timestep.
        
        Diese Funktion ersetzt die stochastische Timestep-Auswahl (ob nun "sigmoid", 
        "uniform" oder die "else"-Branch) durch einen deterministischen Input.
        """
        
        bsz = latents.shape[0]
        num_timesteps = 1000 # z.B. 1000

        # --- Start der Modifikation ---
        
        # Wir ersetzen den gesamten if/elif/else-Block aus der Originalfunktion
        # durch diese deterministische Berechnung:
        
        # 1. Konvertiere den Ziel-Integer (z.B. 500) in einen Float (500.0)
        timesteps_float = float(target_timestep)
        
        # 2. Erstelle einen Tensor, der diesen Timestep für den gesamten Batch hält
        # (z.B. tensor([500.0, 500.0, ...]))
        timesteps = torch.full((bsz,), timesteps_float, device=device, dtype=dtype)
        
        # 3. Berechne sigmas direkt aus den Timesteps.
        # Dies ist die Umkehrung der Logik: timesteps = sigmas * num_timesteps
        # (z.B. 500.0 / 1000.0 = 0.5)
        sigmas = timesteps / num_timesteps

        # --- Ende der Modifikation ---

        # Der restliche Code ist identisch mit Ihrer Originalfunktion.
        # Er beschreibt die "Physik", wie das Rauschen angewendet wird.
        
        # Broadcast sigmas auf die 4D-Form (Batch, Channels, Höhe, Breite)
        sigmas_4d = sigmas.view(-1, 1, 1, 1)

        # Wende das Rauschen an
        if self.args.ip_noise_gamma:
            # ... (Ihre spezielle ip_noise_gamma Logik)
            xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
            if self.args.ip_noise_gamma_random_strength:
                ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * self.args.ip_noise_gamma
            else:
                ip_noise_gamma = self.args.ip_noise_gamma
            noisy_model_input = (1.0 - sigmas_4d) * latents + sigmas_4d * (noise + ip_noise_gamma * xi)
        else:
            # Standard-Pfad: (1-sigma) * x + sigma * noise
            noisy_model_input = (1.0 - sigmas_4d) * latents + sigmas_4d * noise

        # Wir geben den 1D-Timestep-Tensor und den 4D-Sigma-Tensor zurück
        # (oder 1D sigmas, je nachdem, was Ihr Code erwartet - passen Sie sigmas/sigmas_4d an)
        return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas_4d.to(dtype)
    
    def get_svdlayer_gradients(self, calibration_dataloader: DataLoader, save_model_steps,  device: Literal["cuda:0", "cpu"] = "cuda:0", log_file: Optional[str] = None, *args, **kwargs):
        grasp_layer_grads = {}
        grasp_layer_names = self.check_exists_grasp_layer()
        for step, batch in enumerate(calibration_dataloader):
          
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(self.accelerator.device, dtype=self.weight_dtype)
            else:
                with torch.no_grad():
                    # encode images to latents. images are [-1, 1]
                    latents = self.ae.encode(batch["images"].to(self.ae.dtype)).to(self.accelerator.device, dtype=self.weight_dtype)

                
                if torch.any(torch.isnan(latents)):
                    self.accelerator.print("NaN found in latents, replacing with zeros")
                    latents = torch.nan_to_num(latents, 0, out=latents)
            text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
            if text_encoder_outputs_list is not None:
                text_encoder_conds = text_encoder_outputs_list
            else:
                # not cached or training, so get from text encoders
                tokens_and_masks = batch["input_ids_list"]
                with torch.no_grad():
                    input_ids = [ids.to(self.accelerator.device) for ids in batch["input_ids_list"]]
                    text_encoder_conds = self.text_encoding_strategy.encode_tokens(
                        self.flux_tokenize_strategy, [self.clip_l, self.t5xxl], input_ids, self.args.apply_t5_attn_mask
                    )
                    if self.args.full_fp16:
                        text_encoder_conds = [c.to(self.weight_dtype) for c in text_encoder_conds]

            # TODO support some features for noise implemented in get_noise_noisy_latents_and_timesteps

            # Sample noise that we'll add to the latents
            max_loss = 0
            for timestep in tqdm(range(1,1000)):
                self.model.zero_grad()
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # get noisy model input and timesteps
                noisy_model_input, timesteps, sigmas = self.get_noisy_model_input_at_timestep(
                    timestep, latents, noise, self.accelerator.device, self.weight_dtype
                )

                # pack latents and get img_ids
                packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
                packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
                img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=self.accelerator.device)

                # get guidance: ensure args.guidance_scale is float
                guidance_vec = torch.full((bsz,), float(self.args.guidance_scale), device=self.accelerator.device)

                # call model
                l_pooled, t5_out, txt_ids, t5_attn_mask = text_encoder_conds
                if not self.args.apply_t5_attn_mask:
                    t5_attn_mask = None

                with self.accelerator.autocast():
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    model_pred = self.model(
                        img=packed_noisy_model_input,
                        img_ids=img_ids,
                        txt=t5_out,
                        txt_ids=txt_ids,
                        y=l_pooled,
                        timesteps=timesteps / 1000,
                        guidance=guidance_vec,
                        txt_attention_mask=t5_attn_mask,
                        )   

                # unpack latents
                model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

                # apply model prediction type
                model_pred, weighting = flux_train_utils.apply_model_prediction_type(self.args, model_pred, noisy_model_input, sigmas)
                
                # flow matching loss: this is different from SD3
                target = noise - latents
                # calculate loss
                huber_c = train_util.get_huber_threshold_if_needed(self.args, timesteps, self.noise_scheduler)
                loss = train_util.conditional_loss(model_pred.float(), target.float(), self.args.loss_type, "none", huber_c)
                if weighting is not None:    
                    loss = loss * weighting
                if self.args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                loss = loss * loss_weights
                loss = loss.mean()
                
                max_loss = max(max_loss, loss)
                if loss < max_loss*0.05:
                    break        

                # backward
                self.accelerator.backward(loss)
               
                for grasp_layer_name in grasp_layer_names:
                    module: GRASPLayer = self.model.get_submodule(grasp_layer_name)
                    if not module:
                        raise ValueError("module can not found")
                    grad_copy = module.S.grad.detach().clone()
                    if grasp_layer_name not in grasp_layer_grads:
                        grasp_layer_grads[grasp_layer_name] = torch.abs(grad_copy)
                    else:
                        grasp_layer_grads[grasp_layer_name] += torch.abs(grad_copy)
                
     
            if step >= save_model_steps:
                break

        self.grasp_layer_grads = grasp_layer_grads

        return grasp_layer_grads


    def dynamic_svd_selection(
            self,
            grasp_layer_grads: dict,
            metric: Literal["gradient", "taylor"] = "taylor",
            compression_ratio: Optional[float] = None,
            threshold_ratio: Optional[float] = None,
            verbose: Optional[bool] = False,
            log_file: Optional[str] = None
        ):
        if not grasp_layer_grads:
            grasp_layer_grads = self.grasp_layer_grads
            raise ValueError("gradients of grasp_layer should be given, but got None")

        indices_dict = {}

        for grasp_layer_name, grasp_layer_grad in grasp_layer_grads.items():
            grasp_layer: GRASPLayer = self.model.get_submodule(grasp_layer_name)
            S = grasp_layer.S

            if metric == "gradient":
                svd_importance: torch.Tensor = torch.abs(grasp_layer_grad)
            elif metric == "taylor":
                svd_importance: torch.Tensor = torch.abs(grasp_layer_grad * S)
            else:
                raise RuntimeError(f"{metric} not support")

            if grasp_layer.compression_ratio is not None:
                compression_ratio = grasp_layer.compression_ratio

            if compression_ratio is not None:            
                k = self.compute_preserve_rank(grasp_layer, compression_ratio=compression_ratio)
                _, indices = torch.topk(svd_importance, k=k)
            else:
                assert threshold_ratio, "Please provide Taylor threshold to select rank adaptively"
                indices = adaptive_rank_selection(svd_importance_list=svd_importance, target_ratio=threshold_ratio)
            indices_dict[grasp_layer_name] = indices
            self.grasp_values_dict[grasp_layer_name] = {}
            self.grasp_values_dict[grasp_layer_name]["svd_importance"] = torch.round(svd_importance.cpu(), decimals=3).tolist()
            self.grasp_values_dict[grasp_layer_name]["svd_value"] = torch.round(S.data.cpu(), decimals=3).tolist()

        self.indices_dict = indices_dict
        return indices_dict
    
    
    def compile_grasp_model(
        self,
        indices_dict: Optional[dict] = None,
        merge: Optional[bool] = False,
        sigma_fuse: Literal["UV", "U", "V"] = "UV",
        device: Literal["cpu", "cuda"] = "cuda",
        log_file: Optional[str] = None
    ):
        if indices_dict is None:
            indices_dict = self.indices_dict

        rank_dict = {}

        for grasp_layer_name, indices in indices_dict.items():
            grasp_layer: GRASPLayer = self.model.get_submodule(grasp_layer_name)

            S = grasp_layer.S[indices]
            U = grasp_layer.U[:, indices]
            Vh = grasp_layer.Vh[indices, :]
            bias = grasp_layer.bias

            rank_dict[grasp_layer_name] = S.shape[0]

            if merge:
                in_features = Vh.shape[1]
                out_features = U.shape[0]
                self._set_module(self.model, grasp_layer_name, nn.Linear(in_features=in_features, out_features=out_features, bias=True if bias is not None else False))
                linear_layer: nn.Linear = self.model.get_submodule(grasp_layer_name)

                # re-initialize linear weight and bias
                W_compressed = torch.mm(U, torch.mm(torch.diag(S), Vh))
                linear_layer.weight.data = W_compressed

                if bias is not None:
                    linear_layer.bias = bias
                
                linear_layer.requires_grad_(False)
            else:
                self._set_module(self.model, grasp_layer_name, SVDLinear(U=U, S=S, Vh=Vh, bias=bias, sigma_fuse=sigma_fuse))
                svd_linear_layer: SVDLinear = self.model.get_submodule(grasp_layer_name)
                svd_linear_layer.requires_grad_(False)
            
            del grasp_layer
            if "cuda" in device:
                torch.cuda.empty_cache()
        return