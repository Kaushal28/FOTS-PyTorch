import os
from time import time

import torch

import numpy as np

from tqdm import tqdm

from utils import TranscriptEncoder, classes

class Train:
    """
    Trainer class which defines model training and evaluation methods.
    """

    def __init__(self, model, train_iterator, valid_iterator, loss, metric, optimizer, lr_scheduler, config):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.transcript_encoder = TranscriptEncoder(classes)
        self.metric = metric
        self.epochs = config["epochs"]
        self.loss = loss
        self.config = config

        self.model.to(self.device)

    def _eval_metrics(self, y_pred, y_true):
        """
        Calculate evaluation metrics given predictions and ground truths.
        """
        precious, recall, hmean = self.metric(y_pred, y_true)
        return np.array([precious, recall, hmean])

    def train_epoch(self, epoch):
        """Train a single epoch."""
        self.model.train()
        epoch_loss, total_metrics = 0, np.zeros(3)

        for i, batch in tqdm(enumerate(self.train_iterator), total=len(self.train_iterator), position=0, leave=True):
            # image_paths, images, bboxs, transcripts, score_map, geo_map, mapping = batch
            images, score_map, geo_map, training_mask = batch

            images = images.to(self.device)
            score_map = score_map.to(self.device)
            geo_map = geo_map.to(self.device)
            training_mask = training_mask.to(self.device)

            self.optimizer.zero_grad()
            
            # Forward pass
            # pred_score_map, pred_geo_map, pred_recog, pred_boxes, pred_mapping, indices = self.model(images, bboxs, mapping)

            # Forward pass
            pred_score_map, pred_geo_map = self.model(images)

            # Calculate loss
            loss = self.loss(score_map, pred_score_map, geo_map, pred_geo_map, training_mask)
            # Backward pass
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            # transcripts = transcripts[indices]
            # pred_boxes = pred_boxes[indices]
            # pred_mapping = mapping[indices]
            # pred_fns = [image_paths[i] for i in pred_mapping]

            # labels, label_lengths = self.transcript_encoder.encode(transcripts.tolist())
            # labels, label_lengths = labels.to(self.device), label_lengths.to(self.device)
            # recog = (labels, label_lengths)

            # # Calculate loss
            # loss = self.loss(score_map, pred_score_map, geo_map, pred_geo_map, recog, pred_recog)
            # # Backward pass
            # loss.backward()
            # self.optimizer.step()

            # epoch_loss += loss.item()
            # print(f'### LOSS: {loss.item()}')
            # pred_transcripts = []
            # if len(pred_mapping) > 0:
            #     pred, lengths = pred_recog
            #     _, pred = pred.max(2)
            #     for idx in range(lengths.numel()):
            #         l = lengths[idx]
            #         p = pred[:l, idx]
            #         txt = self.transcript_encoder.decode(p, l)
            #         pred_transcripts.append(txt)
            #     pred_transcripts = np.array(pred_transcripts)

            # total_metrics += self._eval_metrics(
            #     (pred_boxes, pred_transcripts, pred_fns),
            #     (bboxs, transcripts, pred_fns)
            # )
        
        return (
            epoch_loss / len(self.train_iterator)
        )
        # return (
        #     epoch_loss / len(self.train_iterator),
        #     total_metrics[0] / len(self.train_iterator),  # precision
        #     total_metrics[1] / len(self.train_iterator),  # recall
        #     total_metrics[2] / len(self.train_iterator)  # f1-score
        # )

    def eval_epoch(self):
        """Validate after training a single epoch."""
        self.model.eval()
        # total_metrics = np.zeros(3)

        val_loss = 0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_iterator), total=len(self.valid_iterator), position=0, leave=True):
                # image_paths, images, bboxs, transcripts, score_map, geo_map, mapping = batch
                images, score_map, geo_map, training_mask = batch

                images = images.to(self.device)
                score_map = score_map.to(self.device)
                geo_map = geo_map.to(self.device)
                training_mask = training_mask.to(self.device)


                # Forward pass
                # pred_score_map, pred_geo_map, pred_recog, pred_boxes, pred_mapping, indices = self.model(images, bboxs, mapping)

                # Forward pass
                pred_score_map, pred_geo_map = self.model(images)

                # Calculate loss
                val_loss += self.loss(score_map, pred_score_map, geo_map, pred_geo_map, training_mask)

                # pred_transcripts = []
                # pred_fns = []
                # if len(pred_mapping) > 0:
                #     pred_mapping = pred_mapping[indices]
                #     pred_boxes = pred_boxes[indices]
                #     pred_fns = [image_paths[i] for i in pred_mapping]

                #     pred, lengths = pred_recog
                #     _, pred = pred.max(2)
                #     for i in range(lengths.numel()):
                #         l = lengths[i]
                #         p = pred[:l, i]
                #         t = self.transcript_encoder.decode(p, l)
                #         pred_transcripts.append(t)
                #     pred_transcripts = np.array(pred_transcripts)

                # gt_fns = [image_paths[i] for i in mapping]
                # total_metrics += self._eval_metrics((pred_boxes, pred_transcripts, pred_fns),
                #                                         (bboxs, transcripts, gt_fns))

        return val_loss / len(self.valid_iterator)
        
        # return (
        #     total_metrics[0] / len(self.train_iterator),  # precision
        #     total_metrics[1] / len(self.train_iterator),  # recall
        #     total_metrics[2] / len(self.train_iterator)  # f1-score
        # )
    
    @staticmethod
    def epoch_time(start_time, end_time):
        """Measure single epoch time based on epoch's start and end times."""
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self):
        """Train the model for given numner of epochs."""
        for epoch in range(self.epochs):
            # Epoch start time
            start_time = time()

            # Train
            # loss, train_precision, train_recall, train_f1 = self.train_epoch(epoch)
            train_loss = self.train_epoch(epoch)
            # Evaluate
            # val_precision, val_recall, val_f1 = 
            val_loss = self.eval_epoch()

            self.lr_scheduler.step(val_loss)

            # Epoch start time
            end_time = time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            # Save the model at "save_freq" and at last epoch
            if ((epoch+1) % self.config["model_save_freq"]) == 0 or epoch+1 == self.epochs:
                if not os.path.isdir(self.config["model_save_path"]):
                    os.makedirs(self.config["model_save_path"], exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.config["model_save_path"], f'FOTS_epoch{epoch+1}.pt')
                )

            # Log the training progress per epoch
            # print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            # print(f'\t Train Loss: {loss:.3f} | Train Precision: {train_precision:7.3f} | Train Recall: {train_recall:7.3f} | Train F1: {train_f1:7.3f}')
            # print(f'\t Val. Precision: {val_precision:7.3f} | Val. Recall: {val_recall:7.3f} | Val F1: {val_f1:7.3f}')
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train Loss: {train_loss:.3f}')
            print(f'\t Val. Loss: {val_loss:.3f}')
