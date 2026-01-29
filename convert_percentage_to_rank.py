def compute_svd_rank_for_compression(n: int, m: int, ratio_x: float) -> int:
    """
    Berechnet den Ziel-Rang (r') für eine SVD-basierte Kompression, 
    um die Parameterzahl um ratio_x zu reduzieren.
    
    Args:
        n (int): Eingabedimension (Zeilen der Matrix W).
        m (int): Ausgabedimension (Spalten der Matrix W).
        ratio_x (float): Gewünschtes Reduktionsverhältnis der Parameter (z.B. 0.6 für 60% Reduktion).

    Returns:
        int: Der Ziel-Rang r' (aufgerundet).
    """
    P_orig = n * m
    P_target = P_orig * (1.0 - ratio_x)
    r_float = P_target / (n + m)
    r_prime = max(1, int(np.ceil(r_float)))
    if ratio_x >= 1.0:
        return 0
    r_prime = min(r_prime, min(n, m))
    return int(r_prime)