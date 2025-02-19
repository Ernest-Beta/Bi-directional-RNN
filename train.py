import torch
import torch.optim as optim
from tqdm import tqdm
from BiRNN.StackedBiNN import StackedBiRNN  #fortosi tou montelou
from BiRNN.utils import load_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
#fortosi twn dedomenwn
print("Loading dataset...")
train_loader, dev_loader, vocab_size, pretrained_embedding_tensor = load_data()
print(f"Dataset loaded! Vocabulary Size: {vocab_size}")

#orismos tou device (cpu h gpu)
print("Initializing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#arxikopoihsh tou montelou
model = StackedBiRNN(
    vocab_size=vocab_size,
    embed_dim=300,
    hidden_dim=256,
    num_layers=2,
    num_classes=2,
    pretrained_embeddings=pretrained_embedding_tensor,   #fortosi twn embeddings
    freeze_embeddings=True
).to(device)  #metafora tou montelou sto device

print("Model initialized!")

#orismos sunartisis apoleias kai optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
#synartisi axiologisis tou montelou
def evaluate_model(model, test_loader, device, criterion):
    model.eval()  
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():  #apagoreuoume to autograd gia axiologisi
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  #metafora dedomenwn sto device
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()  #pairnoume tis provlepseis
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)  #mesos oros apoleias
    return avg_loss, all_preds, all_labels
#synartisi ekpaideusis tou montelou me early stopping
def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=10, patience=3):
    print(f"Using device: {device}")  #ektuposi tou xristimopoioumenou device
    print("Training started!")
    
    train_losses = []
    dev_losses = []
    
    best_dev_loss = float('inf')
    patience_counter = 0  #metritis epochs xwris veltiwsi
    
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs} - Training...")
        model.train()
        train_loss_total = 0.0
        batch_count = 0  #metritis batches
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  #metafora sto device

            optimizer.zero_grad()
            outputs = model(inputs)  
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            batch_count += 1

            #ektuposi anamesa stis epochs gia elegxo proodou
            if batch_count % 50 == 0:
                print(f"[Batch {batch_count}] Loss: {loss.item():.4f}")

        #upologismos mesou orou apoleias ekpaideusis
        train_loss_avg = train_loss_total / len(train_loader)
        train_losses.append(train_loss_avg)  
        print(f" Epoch [{epoch+1}/{num_epochs}] - Average Train Loss: {train_loss_avg:.4f}")

        #axiologisi sto development set
        model.eval()  
        dev_loss, _, _ = evaluate_model(model, dev_loader, device, criterion)   
        dev_losses.append(dev_loss) 
        
        #elegxos an iparxei veltiwsi sti xamili apoleia
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            model_path = f"best_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved with improved dev loss: {dev_loss:.4f}")
        else:
            patience_counter += 1
        
        #early stopping an den iparxei veltiwsi gia xroniko diastima "patience"
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
    
    print(" Training complete!")
    epochs = np.arange(1, len(train_losses) + 1)

    #dhmiourgia smoother curves me cubic spline interpolation
    train_interp = interp1d(epochs, train_losses, kind="cubic")
    dev_interp = interp1d(epochs, dev_losses, kind="cubic")

    epochs_smooth = np.linspace(epochs.min(), epochs.max(), num=200)

    train_losses_smooth = train_interp(epochs_smooth)
    dev_losses_smooth = dev_interp(epochs_smooth)

    #diagramma tis ekpaideusis kai tou development loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_smooth, train_losses_smooth, label="Train Loss (Smoothed)",color = 'red', linestyle="--")
    plt.plot(epochs_smooth, dev_losses_smooth, label="Dev Loss (Smoothed)",color = 'green', linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Development Loss Curves (Smoothed)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return model


#ekpaideusi tou montelou
trained_model = train_model(model, train_loader, dev_loader, criterion, optimizer)

#apothikeusi tou telikou montelou
print("Saving trained model...")
torch.save(trained_model.state_dict(), "BiRNN/best_model.pth")
print("Model saved successfully!")
