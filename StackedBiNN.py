import torch
import torch.nn as nn

class StackedBiRNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embed_dim, 
                 hidden_dim, 
                 num_layers, 
                 num_classes, 
                 pretrained_embeddings=None, 
                 freeze_embeddings=False):
        super(StackedBiRNN, self).__init__()
        
        #fortosi twn pretrained embeddings an dinontai
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings.float(),  #metatrepei ta embeddings se float dtype
                freeze=freeze_embeddings #an freeze=True, ta embeddings den tha ekpaideftoun
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim) #an den uparxoun, dimiourgei nea embeddings

        #orismos tou bidirectional LSTM (an num_layers > 1, exei stack LSTMs)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            #h eisodos tha exei morfi [batch_size, seq_length, embed_dim]
            batch_first=True,
            bidirectional=True, #duadromiko LSTM gia kaluteri anaparastasi
            dropout = 0.3 #dropout gia regularization
        )
        
        #teliki fully connected layer, me 2 * hidden_dim epeidi einai bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_length]
        embedded = self.embedding(x)  #to x metatrepetai se embeddings [batch_size, seq_length, embed_dim]
        
        #pernaei apo to LSTM
        lstm_out, _ = self.lstm(embedded)  
        #lstm_out: [batch_size, seq_length, 2 * hidden_dim] (bidirectional)

        #global max pooling sto xroniko dimension
        out_pooled = torch.max(lstm_out, dim=1)[0]  # [batch_size, 2 * hidden_dim]
        
        #teliki provlepsi apo to fully connected layer
        logits = self.fc(out_pooled)  # [batch_size, num_classes]
        
        return logits
