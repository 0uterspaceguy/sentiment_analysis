from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
import torch
from tqdm import tqdm

"""
Bert transformer model with linear layer
for classification
"""
class Bert(torch.nn.Module):
    def __init__(self, path2model, n_classes=3):
        super(Bert, self).__init__()
        self.model = AutoModel.from_pretrained(path2model)
        self.fc = torch.nn.Linear(768, n_classes)

    def forward(self, ids, masks):
        emb = self.model(input_ids=ids, attention_mask=masks)[1]
        return self.fc(emb)


"""
A wrapper for bert model
"""

class Model():
    def __init__(self, hug_path2model, local_path2model=False):
        """
        Load model from huggingface server, then load local model if we need
        """
        self.bert = Bert(hug_path2model)
        if local_path2model:
            self.load_model(local_path2model)

    def load_model(self, path2model):
        """
        Function for load model:

        Args:
            path2model (str): path to local model weights
        """
        checkpoint = torch.load(path2model, map_location='cpu')
        self.bert.load_state_dict(checkpoint['model_state_dict'])

    def save_model(self, path2model):
        """
        Function for save model:

        Args:
            path2model (str): path to save model weights directory
        """
        torch.save({"model_state_dict": self.bert.state_dict()}, path2model)

    def train(self, train_dataloader, val_dataloader, epochs=10, lr=2e-5, warmup=100, validate_steps=1000,
              loss_print=200):
        """
           Trainig function:

           Args:
               train_dataloader (Dataloader.Dataloader): training dataloader
               val_dataloader (Dataloader.Dataloader): validation dataloader
               epochs (int): number of training loops
               lr (float): initial learning rate
               warmup (int): number of warmup steps
               validate_steps (int): number steps before validation
               loss_print (int): number steps before print loss
        """

        param_optimizer = list(self.bert.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, \
                                                    num_training_steps=len(train_dataloader) * epochs)
        criterion = torch.nn.CrossEntropyLoss().to('cuda')

        self.bert.to('cuda').train()
        self.f1 = [0]
        losses = []

        for epoch in range(epochs):
            print(f"Epoch: {epoch + 1}")
            for step, batch in enumerate(train_dataloader, start=1):

                if step % loss_print == 0:
                    print(f"Loss: {sum(losses) / len(losses)}")
                    losses.clear()

                if step % validate_steps == 0:
                    self.test(val_dataloader)
                    self.bert.train()

                batch = tuple(torch.tensor(t).to('cuda') for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                logits = self.bert(b_input_ids, b_input_mask)
                loss = criterion(logits, b_labels.argmax(axis=1))
                losses.append(loss)

                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    def test(self, dataloader, with_save = True):
        """
            Testing function:

            Args:
                dataloader (Dataloader.Dataloader): testing dataloader
                with_save (bool): save model or not
         """
        self.bert.eval().to('cuda')
        predictions = []
        targets = []

        print("Validation: ")

        for step, batch in enumerate(tqdm(dataloader)):
            batch = tuple(torch.tensor(t).to('cuda') for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                logits = self.bert(b_input_ids, b_input_mask)

                targets.extend(b_labels.argmax(axis=1).to('cpu').tolist())
                predictions.extend(logits.argmax(axis=1).to('cpu').tolist())

        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')

        print('Accuracy score: {:.2f}'.format(accuracy))
        print('F1 score: {:.2f}'.format(f1))

        if with_save:
            if f1 > self.f1[-1]:
                self.save_model('model.pth')
            self.f1.append(f1)

    def predict(self, input_ids, input_mask, device='cpu'):
        """
        Function makes prediction for input data:

        Args:
            input_ids (np.ndarray): input ids for model
            input_mask (np.ndarray): input mask for model

        Returns:
            (list): list of labels
        """
        input_ids, input_mask = tuple(torch.tensor(t).to(device) for t in [input_ids, input_mask])
        self.bert.to(device).eval()
        with torch.no_grad():
            logits = self.bert(input_ids, input_mask)
            return logits.argmax(axis=1).tolist()
