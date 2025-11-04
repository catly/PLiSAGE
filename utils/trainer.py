from utils.metrics import classification_metrics

class ClassificationTrainer:
    """A class to encapsulate the training and evaluation process for classification tasks."""
    def __init__(self, model, optimizer, scheduler, device, train_loader, val_loader, test_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.best_auc = 0.0

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.training.num_epochs}')):
            protein_graph, xyz, curvature, dists, atom_type_sel, drug_graph, label = set_gpu(batch, self.device)
            points = self.config.train_transforms(xyz)

            out = self.model(protein_graph, points, curvature, dists, atom_type_sel, drug_graph)
            loss = F.binary_cross_entropy(out, label)
            loss = loss / self.config.training.accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * self.config.training.accumulation_steps

            if (i + 1) % self.config.training.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(self.train_loader)
        logging.info(f"Epoch {epoch + 1} | Average Train Loss: {avg_loss:.4f}")

    def _evaluate(self, loader):
        self.model.eval()
        predictions = []
        labels = []
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                protein_graph, xyz, curvature, dists, atom_type_sel, drug_graph, label = set_gpu(batch, self.device)
                
                pred = self.model(protein_graph, xyz, curvature, dists, atom_type_sel, drug_graph)
                loss = F.binary_cross_entropy(pred, label)
                total_loss += loss.item()

                predictions.extend(pred.cpu().tolist())
                labels.extend(label.cpu().tolist())
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(loader)
        return avg_loss, labels, predictions

    def train(self):
        logging.info("Starting training...")
        for epoch in range(self.config.training.num_epochs):
            self._train_epoch(epoch)

            # Validation
            avg_val_loss, val_labels, val_preds = self._evaluate(self.val_loader)
            self.scheduler.step(avg_val_loss)
            val_metrics = classification_metrics(val_labels, val_preds)
            logging.info(f"Validation | Loss: {avg_val_loss:.4f} | AUC: {val_metrics['roc_auc']:.4f} | AUPRC: {val_metrics['auprc']:.4f}")

            # Save best model and run test
            if val_metrics['roc_auc'] > self.best_auc:
                self.best_auc = val_metrics['roc_auc']
                logging.info(f"******** New best model found! Saving to {self.config.training.checkpoint_dir} ********")
                if not os.path.exists(self.config.training.checkpoint_dir):
                    os.makedirs(self.config.training.checkpoint_dir)
                torch.save(self.model.state_dict(), os.path.join(self.config.training.checkpoint_dir, 'best_model.pt'))

                # Run test on new best model, using the optimal threshold from validation
                optimal_thresh = val_metrics['optimal_threshold']
                avg_test_loss, test_labels, test_preds = self._evaluate(self.test_loader)
                test_metrics = classification_metrics(test_labels, test_preds, threshold=optimal_thresh)
                
                log_msg = (
                    f"Test Results (Thresh={optimal_thresh:.4f}) | Loss: {avg_test_loss:.4f} | AUC: {test_metrics['roc_auc']:.4f} | "
                    f"AUPRC: {test_metrics['auprc']:.4f} | F1: {test_metrics['f1_score']:.4f} | Acc: {test_metrics['accuracy']:.4f}"
                )
                logging.info(log_msg)
                with open(os.path.join(self.config.training.checkpoint_dir, "best_metrics_log.txt"), "w") as f:
                    f.write(log_msg)