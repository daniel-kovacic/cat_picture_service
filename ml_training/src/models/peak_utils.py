import torch
import torch.nn.functional as F


class PeakUtils:

    @staticmethod
    def find_multiple_most_likely_normalized_coord(
            hm: torch.Tensor,
            threshold: float = 0.3,
            pool_kernel: int = 5,
            max_peaks_per_channel: int = 10,
    ):
        C, H, W = hm.shape

        hm_unsq = hm.unsqueeze(0)
        pooled = F.max_pool2d(
            hm_unsq,
            kernel_size=pool_kernel,
            stride=1,
            padding=pool_kernel // 2,
        )

        is_peak = (hm_unsq == pooled) & (hm_unsq > threshold)
        is_peak = is_peak.squeeze(0)

        results = []

        for c in range(C):
            peak_mask = is_peak[c]
            if peak_mask.sum() == 0:
                results.append(torch.empty((0, 3), device=hm.device))
                continue

            ys, xs = torch.where(peak_mask)
            scores = hm[c, ys, xs]

            order = torch.argsort(scores, descending=True)
            order = order[:max_peaks_per_channel]

            xs = xs[order].float()
            ys = ys[order].float()
            scores = scores[order]

            x_norm = xs / (W - 1)
            y_norm = ys / (H - 1)

            coords = torch.stack([x_norm, y_norm, scores], dim=-1)
            results.append(coords)

        return results

    @staticmethod
    def group_landmarks_list(
            nose_peaks,
            left_eye_peaks,
            right_eye_peaks,
            left_ear_peaks,
            right_ear_peaks,
            max_nose_to_landmark_scale: float = 1.5,
    ):
        cats = []

        if nose_peaks is None or len(nose_peaks) == 0:
            return cats

        device = nose_peaks.device

        def select_best(candidate_peaks, nose_xy, eye_dist):
            if candidate_peaks is None or len(candidate_peaks) == 0:
                return None

            dists = torch.norm(candidate_peaks[:, :2] - nose_xy, dim=1)
            norm_dists = dists / (eye_dist + 1e-6)

            scores = candidate_peaks[:, 2] - 1.5 * norm_dists
            best_idx = torch.argmax(scores)

            if norm_dists[best_idx] > max_nose_to_landmark_scale:
                return None

            return candidate_peaks[best_idx]

        for n in nose_peaks:
            n_xy = n[:2]

            if len(left_eye_peaks) == 0 or len(right_eye_peaks) == 0:
                continue

            dl = torch.norm(left_eye_peaks[:, :2] - n_xy, dim=1)
            dr = torch.norm(right_eye_peaks[:, :2] - n_xy, dim=1)

            li = torch.argmin(dl)
            ri = torch.argmin(dr)

            left_eye = left_eye_peaks[li]
            right_eye = right_eye_peaks[ri]

            if left_eye[0] >= right_eye[0]:
                continue

            eye_dist = torch.norm(left_eye[:2] - right_eye[:2])
            if eye_dist < 1e-4:
                continue

            cat = torch.full((9, 3), float("nan"), device=device)

            cat[0] = left_eye
            cat[1] = right_eye
            cat[2] = n
            for k in range(3):
                ear = select_best(left_ear_peaks[k], n_xy, eye_dist)
                if ear is not None:
                    cat[3 + k] = ear

            for k in range(3):
                ear = select_best(right_ear_peaks[k], n_xy, eye_dist)
                if ear is not None:
                    cat[6 + k] = ear

            cats.append(cat)

        return cats

    @staticmethod
    def extract_cats_from_heatmap(
            hm: torch.Tensor,
            peak_threshold: float = 0.3,
            pool_kernel: int = 5,
            max_peaks_per_channel: int = 10,
            max_nose_to_landmark_scale: float = 1.5,
    ):

        peak_list = PeakUtils.find_multiple_most_likely_normalized_coord(
            hm,
            threshold=peak_threshold,
            pool_kernel=pool_kernel,
            max_peaks_per_channel=max_peaks_per_channel,
        )

        if len(peak_list) < 9:
            return []

        left_eye_peaks = peak_list[0]
        right_eye_peaks = peak_list[1]
        nose_peaks = peak_list[2]

        left_ear_peaks = peak_list[3:6]

        right_ear_peaks = peak_list[6:]

        cats = PeakUtils.group_landmarks_list(
            nose_peaks=nose_peaks,
            left_eye_peaks=left_eye_peaks,
            right_eye_peaks=right_eye_peaks,
            left_ear_peaks=left_ear_peaks,
            right_ear_peaks=right_ear_peaks,
            max_nose_to_landmark_scale=max_nose_to_landmark_scale,
        )

        return cats
