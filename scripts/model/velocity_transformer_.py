import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from transformers import (
    AutoConfig,
    AutoModel,
)


class VelocityFormer(pl.LightningModule):
        
    def __init__(self, cfg, lr=1e-5):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率
        super().__init__()
        model_name = cfg.pretrained_model

        # BERTのロード
        self.save_hyperparameters() 
        model_config = AutoConfig.from_pretrained(model_name)  # load config from hugging face model
        model_config.input_size = cfg.input_size
        num_labels = cfg.num_labels
        n_embd = model_config.hidden_size

        self.bert_sc= AutoModel.from_config(model_config)
        self.linear = torch.nn.Linear(n_embd, num_labels)
        self.criterion = RMSELoss()

    def forward(self, data: torch.Tensor):
        bert_output = self.bert_sc(input_ids = data)
        last_hidden_state = bert_output.last_hidden_state
        averaged_hidden_state, _ = last_hidden_state.max(1) # max_pooling
        logits = self.linear(averaged_hidden_state)
        return logits

    # 学習データの損失を出力する関数
    def training_step(self, batch, batch_idx):
        logits = self(batch["data"])
        output = {"logits": logits}
        loss = self.criterion(batch["label"], logits)
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss

    # テストデータを評価する関数。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('label') # バッチからラベルを取得
        output = self(batch['data'])
        loss = self.criterion(labels, output)
        self.log('loss', loss) # 精度を'accuracy'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

def main():
    MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    model = VelocityFormer(
    MODEL_NAME, num_labels=9, lr=1e-5)

if __name__ == "__main__":
    main()