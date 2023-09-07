import torch
import matplotlib.pyplot as plt
import numpy as np

def inspect_data(loader):
    preview_dim = 20
    fig, axs = plt.subplots(preview_dim, preview_dim, figsize = (20, 20))
    
    for i in range(preview_dim):
        for j in range(preview_dim):
            features, labels = next(iter(loader))
            idx = torch.randint(0, features.shape[0] - 1, (1,))
            axs[i, j].axis("off")
            axs[i, j].imshow(features[idx].reshape(3, 32, 32).permute(1,2,0))

    plt.show()

def draw_loss_graphs(train_loss_seq, test_loss_seq):
	plt.plot(train_loss_seq, label = 'Training loss')
	plt.plot(test_loss_seq, label = 'Test loss')
	plt.legend()

def inspect_misclassified(net, test_loader):
	label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
             5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
	label_dict_inv = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
                 "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}

	fig, axs = plt.subplots(16, 12, figsize = (20, 30))
	with torch.no_grad():
	    for i in range(16):
	        net.eval()
	        features, labels = next(iter(test_loader))
	        
	        outputs = net(features)
	        _, preds = outputs.max(1)
	        misclass = ~preds.eq(labels)
	            
	        input_misclass = features[misclass]
	        preds_misclass = preds[misclass]
	        
	        J = min(input_misclass.shape[0] - 1, 12)
	        init = list(range(input_misclass.shape[0] - 1))
	        idxs = np.random.choice(init, J, replace = False)
	        
	        for j in range(J):
	            idx = idxs[j]
	            img = input_misclass[idx]
	            label = preds_misclass[idx]
	            axs[i, j].imshow(img.reshape(3, 32, 32).permute(1,2,0))
	            axs[i, j].set_title(label_dict[label.item()])
	            axs[i, j].axis("off")

	    plt.show()