"""
Partial Mutual Inductance Simulator for Planar Coils
Based on:
Mahdi Asadi, Amir Musa Abazari,
"A Data-Driven Model to Predict Mutual Inductance Between Planar Coils With Arbitrary Specifications and Positions",
IET Electric Power Applications, 2025.
DOI: 10.1049/elp2.70015
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy.integrate import dblquad

MU_0 = 4.0 * np.pi * 1e-7  # Vacuum permeability (H/m)


class PlanarPolygonCoil:
    """
    Represents an n-sided regular planar polygon coil.
    
    Parameters
    ----------
    ns : int
        Number of sides of the regular polygon (e.g., 4 for square, 6 for hexagon, 8 for octagon).
    nt : int
        Number of turns.
    s : float
        Spacing between adjacent turns in millimeters.
    r_i : float
        Inner circumcircle radius in millimeters (radius of the innermost turn).
    w : float, optional
        Conductor / wire track width in millimeters (default: 0.0).
        If w > 0, the turn pitch is p = s + w.
    """
    def __init__(self, ns: int, nt: int, s: float, r_i: float, w: float = 0.0):
        if ns < 3:
            raise ValueError(f"Number of sides ns must be >= 3, got {ns}")
        if nt < 1:
            raise ValueError(f"Number of turns nt must be >= 1, got {nt}")
        if r_i <= 0:
            raise ValueError(f"Inner radius r_i must be > 0, got {r_i}")
        if s <= 0:
            raise ValueError(f"Spacing s must be > 0, got {s}")

        self.ns = int(ns)
        self.nt = int(nt)
        self.s = float(s)        # mm
        self.r_i = float(r_i)    # mm
        self.w = float(w)        # mm
        self.pitch = self.s + self.w  # mm

        # Outer circumcircle radius (radius of outermost turn)
        self.r_o = self.r_i + (self.nt - 1) * self.pitch  # mm

    def get_turn_radii(self) -> np.ndarray:
        """Returns the circumcircle radius of each turn in meters."""
        indices = np.arange(self.nt)
        radii_mm = self.r_i + indices * self.pitch
        return radii_mm * 1e-3  # convert to meters

    def get_turn_segments(self, turn_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (starts, ends) coordinates in meters in the coil's local frame (z=0)
        for a specific turn (0-indexed).
        
        According to the paper (Fig. 2c):
        A_m = (r * cos(2(m-1)*pi / ns), r * sin(2(m-1)*pi / ns), 0)
        for m = 1, ..., ns.
        """
        r = (self.r_i + turn_idx * self.pitch) * 1e-3  # meters
        m = np.arange(1, self.ns + 1)
        theta = 2.0 * (m - 1) * np.pi / self.ns
        
        # Start points of each side
        starts = np.stack([r * np.cos(theta), r * np.sin(theta), np.zeros(self.ns)], axis=1)
        # End points of each side (connecting vertex m to vertex m+1, closing the polygon)
        ends = np.roll(starts, -1, axis=0)
        return starts, ends

    def get_all_segments(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns all segments across all turns in local frame:
        starts : (N_total, 3)
        ends   : (N_total, 3)
        turn_indices : (N_total,)
        side_indices : (N_total,)
        """
        starts_list = []
        ends_list = []
        turn_idx_list = []
        side_idx_list = []

        for turn in range(self.nt):
            s, e = self.get_turn_segments(turn)
            starts_list.append(s)
            ends_list.append(e)
            turn_idx_list.append(np.full(self.ns, turn))
            side_idx_list.append(np.arange(1, self.ns + 1))

        return (
            np.concatenate(starts_list, axis=0),
            np.concatenate(ends_list, axis=0),
            np.concatenate(turn_idx_list, axis=0),
            np.concatenate(side_idx_list, axis=0)
        )


def build_rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Computes 3D rotation matrix for angular misalignments:
    phi   : rotation angle around X-axis (rad)
    theta : rotation angle around Y-axis (rad)
    psi   : rotation angle around Z-axis (rad)
    
    Using standard composite rotation R = R_x(phi) @ R_y(theta) @ R_z(psi)
    """
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(phi), -np.sin(phi)],
        [0.0, np.sin(phi), np.cos(phi)]
    ])
    ry = np.array([
        [np.cos(theta), 0.0, np.sin(theta)],
        [0.0, 1.0, 0.0],
        [-np.sin(theta), 0.0, np.cos(theta)]
    ])
    rz = np.array([
        [np.cos(psi), -np.sin(psi), 0.0],
        [np.sin(psi), np.cos(psi), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return rx @ ry @ rz


class PartialMutualInductanceCalculator:
    """
    Calculator implementing the partial mutual inductance method
    between two arbitrarily oriented n-sided planar coils.
    """

    @staticmethod
    def calculate_mutual_inductance(
        tx: PlanarPolygonCoil,
        rx: PlanarPolygonCoil,
        translation: Tuple[float, float, float],
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        n_quad: int = 10,
        return_turn_matrix: bool = False
    ) -> Dict[str, any]:
        """
        Calculates the mutual inductance between transmitter (Tx) and receiver (Rx).
        
        Parameters
        ----------
        tx : PlanarPolygonCoil
            Transmitter coil (positioned at origin z=0).
        rx : PlanarPolygonCoil
            Receiver coil.
        translation : (v1, v2, v3)
            Translation vector of Rx relative to Tx in millimeters.
            v1: lateral misalignment along X
            v2: lateral misalignment along Y
            v3: axial distance along Z
        rotation : (phi, theta, psi), optional
            Angular rotation of Rx relative to Tx in radians:
            phi   : around X-axis
            theta : around Y-axis
            psi   : around Z-axis
        n_quad : int, default 10
            Number of Gauss-Legendre quadrature points per segment.
        return_turn_matrix : bool, default False
            If True, calculates and returns the (nt_tx, nt_rx) turn mutual matrix.
            
        Returns
        -------
        result : dict
            {
                'M_H': float (Henries),
                'M_nH': float (nanohenries),
                'M_uH': float (microhenries),
                'turn_matrix_nH': np.ndarray (optional),
                'tx_total_turns': int,
                'rx_total_turns': int,
                'num_segment_pairs': int
            }
        """
        # Convert translation to meters
        trans_m = np.array(translation, dtype=np.float64) * 1e-3
        phi, theta, psi = rotation
        R_mat = build_rotation_matrix(phi, theta, psi)

        # 1. Transmitter segments in meters (at origin z=0)
        tx_starts, tx_ends, tx_turn_idx, tx_side_idx = tx.get_all_segments()
        L_tx = tx_ends - tx_starts  # (N_tx, 3)

        # 2. Receiver segments transformed into global coordinates
        rx_starts_local, rx_ends_local, rx_turn_idx, rx_side_idx = rx.get_all_segments()
        rx_starts = (R_mat @ rx_starts_local.T).T + trans_m  # (N_rx, 3)
        rx_ends = (R_mat @ rx_ends_local.T).T + trans_m      # (N_rx, 3)
        L_rx = rx_ends - rx_starts  # (N_rx, 3)

        # 3. Vector dot products between all segments: (N_tx, N_rx)
        dot_prod = np.dot(L_tx, L_rx.T)

        # 4. Gauss-Legendre quadrature for double integral:
        # \int_0^1 \int_0^1 1 / || r_T(s) - r_R(l) || ds dl
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        s_pts = 0.5 * (nodes + 1.0)
        w_pts = 0.5 * weights
        w_mat = np.outer(w_pts, w_pts)  # (nq, nq)

        # Sample points on all Tx segments: (N_tx, n_quad, 3)
        pts_tx = tx_starts[:, None, :] + s_pts[None, :, None] * L_tx[:, None, :]
        # Sample points on all Rx segments: (N_rx, n_quad, 3)
        pts_rx = rx_starts[:, None, :] + s_pts[None, :, None] * L_rx[:, None, :]

        # Distance between all pairs of quadrature points:
        # diff shape: (N_tx, N_rx, nq, nq, 3)
        diff = pts_tx[:, None, :, None, :] - pts_rx[None, :, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        inv_dist = 1.0 / dist

        # Integrate: contract over quadrature points with weights
        segment_integrals = np.einsum('ijkl,kl->ij', inv_dist, w_mat)  # (N_tx, N_rx)

        # Partial mutual inductance matrix for all segment pairs (Henries)
        factor = MU_0 / (4.0 * np.pi)
        M_segment_pairs = factor * dot_prod * segment_integrals  # (N_tx, N_rx)

        total_M_H = float(np.sum(M_segment_pairs))
        total_M_nH = total_M_H * 1e9
        total_M_uH = total_M_H * 1e6

        result = {
            'M_H': total_M_H,
            'M_nH': total_M_nH,
            'M_uH': total_M_uH,
            'num_segment_pairs': tx_starts.shape[0] * rx_starts.shape[0],
            'tx_total_turns': tx.nt,
            'rx_total_turns': rx.nt
        }

        if return_turn_matrix:
            turn_matrix = np.zeros((tx.nt, rx.nt), dtype=np.float64)
            for t_turn in range(tx.nt):
                mask_tx = (tx_turn_idx == t_turn)
                for r_turn in range(rx.nt):
                    mask_rx = (rx_turn_idx == r_turn)
                    turn_matrix[t_turn, r_turn] = np.sum(M_segment_pairs[mask_tx][:, mask_rx]) * 1e9
            result['turn_matrix_nH'] = turn_matrix

        return result

    @staticmethod
    def calculate_mutual_inductance_equation_1b(
        tx: PlanarPolygonCoil,
        rx: PlanarPolygonCoil,
        translation: Tuple[float, float, float],
        rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        eps: float = 1e-12
    ) -> float:
        """
        Direct implementation of the paper's Equation (1b) and (2a)-(2h):
        
        M_mkij = (mu_0 * Delta / (4 * pi)) * \int_{x_k^R}^{x_{k+1}^R} \int_{x_m^T}^{x_{m+1}^T} (dx_T dx_R / R)
        where:
        Delta = 1 + (Delta_2 / Delta_1) * Delta_4
        Delta_1 = x_{k+1}^R - x_k^R
        Delta_2 = y_{k+1}^R - y_k^R
        Delta_3 = z_{k+1}^R - z_k^R
        Delta_4 = -cot(pi * (2m - 1) / ns_T)
        y_R(x_R) = (Delta_2 / Delta_1) * (x_R - x_k^R) + y_k^R
        z_R(x_R) = (Delta_3 / Delta_1) * (x_R - x_k^R) + z_k^R
        y_T(x_T) = Delta_4 * (x_T - x_m^T) + y_m^T
        z_T = 0
        R = sqrt((x_T - x_R)^2 + (y_T - y_R)^2 + (z_T - z_R)^2)
        
        Summed over m=1..ns_T, k=1..ns_R, i=1..nt_T, j=1..nt_R.
        """
        trans_m = np.array(translation, dtype=np.float64) * 1e-3
        phi, theta, psi = rotation
        R_mat = build_rotation_matrix(phi, theta, psi)

        total_M_H = 0.0
        factor = MU_0 / (4.0 * np.pi)

        # Loop over turns
        for i in range(tx.nt):
            tx_starts, tx_ends = tx.get_turn_segments(i)
            for j in range(rx.nt):
                rx_starts_local, rx_ends_local = rx.get_turn_segments(j)
                rx_starts = (R_mat @ rx_starts_local.T).T + trans_m
                rx_ends = (R_mat @ rx_ends_local.T).T + trans_m

                # Loop over sides of Tx (m = 1..ns_T)
                for m_idx in range(tx.ns):
                    m = m_idx + 1  # 1-indexed
                    A1 = tx_starts[m_idx]
                    A2 = tx_ends[m_idx]
                    
                    dx_T = A2[0] - A1[0]
                    # Equation (2d)
                    cot_val = 1.0 / np.tan(np.pi * (2 * m - 1) / tx.ns)
                    delta4 = -cot_val

                    # Loop over sides of Rx (k = 1..ns_R)
                    for k_idx in range(rx.ns):
                        B1 = rx_starts[k_idx]
                        B2 = rx_ends[k_idx]

                        # Equations (2b), (2c), (2h)
                        delta1 = B2[0] - B1[0]
                        delta2 = B2[1] - B1[1]
                        delta3 = B2[2] - B1[2]

                        # If delta1 or dx_T is nearly zero, the x-projected substitution
                        # has a vertical singularity. We fallback to parametric representation
                        # which is mathematically identical.
                        if abs(delta1) < eps or abs(dx_T) < eps:
                            # Parametric equivalent of the line integral
                            L1 = A2 - A1
                            L2 = B2 - B1
                            dot_val = np.dot(L1, L2)
                            nodes, weights = np.polynomial.legendre.leggauss(8)
                            s = 0.5 * (nodes + 1.0)
                            w = 0.5 * weights
                            p1 = A1[:, None] + L1[:, None] * s[None, :]
                            p2 = B1[:, None] + L2[:, None] * s[None, :]
                            diff = p1[:, :, None] - p2[:, None, :]
                            dist = np.linalg.norm(diff, axis=0)
                            int_val = np.sum(np.outer(w, w) / dist)
                            total_M_H += factor * dot_val * int_val
                        else:
                            # Direct Equation (2a)
                            delta = 1.0 + (delta2 / delta1) * delta4

                            def integrand(x_T, x_R):
                                y_R = (delta2 / delta1) * (x_R - B1[0]) + B1[1]
                                z_R = (delta3 / delta1) * (x_R - B1[0]) + B1[2]
                                y_T = delta4 * (x_T - A1[0]) + A1[1]
                                dist_sq = (x_T - x_R)**2 + (y_T - y_R)**2 + z_R**2
                                return 1.0 / np.sqrt(dist_sq)

                            xR_min, xR_max = min(B1[0], B2[0]), max(B1[0], B2[0])
                            xT_min, xT_max = min(A1[0], A2[0]), max(A1[0], A2[0])

                            sign_T = 1.0 if A2[0] > A1[0] else -1.0
                            sign_R = 1.0 if B2[0] > B1[0] else -1.0

                            val, _ = dblquad(
                                integrand,
                                xR_min, xR_max,
                                xT_min, xT_max,
                                epsabs=1e-12, epsrel=1e-6
                            )
                            M_mkij = factor * delta * val * sign_T * sign_R
                            total_M_H += M_mkij

        return total_M_H * 1e9  # return in nH

    @staticmethod
    def calculate_self_inductance(coil: PlanarPolygonCoil, wire_radius_mm: float = 0.5, n_quad: int = 10) -> Dict[str, float]:
        """
        Calculates the self-inductance of a coil using the partial inductance method (Grover).
        
        Parameters
        ----------
        coil : PlanarPolygonCoil
        wire_radius_mm : float
            Effective radius of the conductor wire in millimeters (default: 0.5 mm).
        n_quad : int
            Quadrature points for mutual segment pairs.
            
        Returns
        -------
        dict with self-inductance in H, nH, uH.
        """
        r_w = wire_radius_mm * 1e-3  # wire radius in meters
        starts, ends, turn_idx, side_idx = coil.get_all_segments()
        N = starts.shape[0]
        L_vec = ends - starts
        lengths = np.linalg.norm(L_vec, axis=1)

        # 1. Self partial inductance of each straight segment (Grover's formula for straight round wire):
        # L_self_i = (mu_0 / 2pi) * l * [ ln(2l / r_w) - 1 + r_w / l ]
        # In high-frequency / thin-wire limit: ln(2l / r_w) - 0.75 or ln(2l / r_w) - 1
        factor_self = MU_0 / (2.0 * np.pi)
        L_segments = factor_self * lengths * (np.log(2.0 * lengths / r_w) - 1.0)
        total_L_self = np.sum(L_segments)

        # 2. Mutual partial inductance between distinct segments of the same coil:
        factor_mut = MU_0 / (4.0 * np.pi)
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        s_pts = 0.5 * (nodes + 1.0)
        w_pts = 0.5 * weights
        w_mat = np.outer(w_pts, w_pts)

        pts = starts[:, None, :] + s_pts[None, :, None] * L_vec[:, None, :]  # (N, nq, 3)
        dot_prod = np.dot(L_vec, L_vec.T)  # (N, N)

        diff = pts[:, None, :, None, :] - pts[None, :, None, :, :]  # (N, N, nq, nq, 3)
        dist = np.linalg.norm(diff, axis=-1)
        
        # Zero out the diagonal (same segment is computed via self-inductance formula above)
        np.fill_diagonal(dot_prod, 0.0)

        # To avoid division by zero on diagonal:
        for i in range(N):
            dist[i, i, :, :] = 1.0

        inv_dist = 1.0 / dist
        for i in range(N):
            inv_dist[i, i, :, :] = 0.0

        integral = np.einsum('ijkl,kl->ij', inv_dist, w_mat)
        total_mutual_segments = factor_mut * np.sum(dot_prod * integral)

        L_total_H = total_L_self + total_mutual_segments
        return {
            'L_H': L_total_H,
            'L_nH': L_total_H * 1e9,
            'L_uH': L_total_H * 1e6
        }
