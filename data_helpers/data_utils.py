import numpy as np

import torch

def icdar_collate(batch):
    """
    Collate function for ICDAR dataset. It receives a batch of ground truths
    and formats it in required format.
    """
    image_paths, img, boxes, transcripts, score_map, geo_map = zip(*batch)
    batch_size = len(score_map)
    images, score_maps, geo_maps = [], [], []

    # convert all numpy arrays to tensors
    for idx in range(batch_size):
        if img[idx] is not None:
            images.append(torch.from_numpy(img[idx]).permute(2, 0, 1))
            score_maps.append(torch.from_numpy(score_map[idx]).permute(2, 0, 1))
            geo_maps.append(torch.from_numpy(geo_map[idx]).permute(2, 0, 1))

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)

    texts, bboxs, mapping = [], [], []
    for idx, (text, bbox) in enumerate(zip(transcripts, boxes)):
        for txt, box in zip(text, bbox):
            mapping.append(idx)
            texts.append(txt)
            bboxs.append(box)

    texts = np.array(texts)
    bboxs = np.stack(bboxs, axis=0)
    bboxs = np.concatenate([bboxs, np.ones((len(bboxs), 1))], axis = 1).astype(np.float32)

    return image_paths, images, bboxs, texts, score_maps, geo_maps, mapping
