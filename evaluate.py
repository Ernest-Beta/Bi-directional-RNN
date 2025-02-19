import torch
from BiRNN.StackedBiNN import StackedBiRNN
from BiRNN.utils import load_test_data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

#orismos tou device (cpu h gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#fortosi twn dedomenwn test
test_loader, vocab_size, pretrained_embedding_tensor = load_test_data()

#den xreiazetai to move toy pretrained_embedding_tensor sto device
#otan kanoume model.to(device), to PyTorch to xeirizetai automata

#fortosi tou montelou
model = StackedBiRNN(
    vocab_size=vocab_size,
    embed_dim=300,  #prepei na tairiazei me to GloVe dimension (100, 200, 300)
    hidden_dim=256,
    num_layers=2,
    num_classes=2,
    pretrained_embeddings=pretrained_embedding_tensor,  
    freeze_embeddings=True
)

#fortosi twn proekpaideumenwn varwn tou montelou
model.load_state_dict(torch.load("best_model_epoch_3.pth", map_location=device))
model.to(device)
model.eval() #to montelo se evaluation mode

#methodos axiologisis tou montelou
def evaluate_model(model, test_loader, device):
    all_preds, all_labels = [], []
    
    with torch.no_grad(): #apagoreuoume to autograd gia axiologisi
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)  #metafora dedomenwn sto device
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()  #pairnoume tis provlepseis
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    #ypologismos precision, recall kai F1-score ana klasi
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0,1])

    #ypologismos macro/micro metrikwn
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")

    #ypologismos accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    #ektupwsi apotelesmatwn
    print("Accuracy:", accuracy)
    print("\nClasswise Precision, Recall, F1-score:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    print("\nMacro Average:")
    print("Precision:", precision_macro)
    print("Recall:", recall_macro)
    print("F1-Score:", f1_macro)

    print("\nMicro Average:")
    print("Precision:", precision_micro)
    print("Recall:", recall_micro)
    print("F1-Score:", f1_micro)

    results = {
        'accuracy': accuracy,
        'classwise': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1-score': f1.tolist()
        },
        'macro': {
            'precision': precision_macro,
            'recall': recall_macro,
            'f1-score': f1_macro
        },
        'micro': {
            'precision': precision_micro,
            'recall': recall_micro,
            'f1-score': f1_micro
        }
    }

    return results

#ektelesi tis axiologisis
results = evaluate_model(model, test_loader, device)
