import torch.optim as optim
from DataLoader import x_train_with_labels, x_test_with_labels
from NetChoice import *
from Trainning import training_loop_with_acc

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)
batch_size = 64

train_loader = torch.utils.data.DataLoader(x_train_with_labels, batch_size=batch_size,
                                           shuffle=True)  # <1>
val_loader = torch.utils.data.DataLoader(x_test_with_labels, batch_size=batch_size,
                                         shuffle=False)

model = NetResDeep(n_blocks=5)  # <2> blocks layer
optimizer = optim.SGD(model.parameters(), lr=3e-3)  # <3>
loss_fn = nn.CrossEntropyLoss()  # <4>

training_loop_with_acc(  # <5>
		n_epochs=100,
		optimizer=optimizer,
		model=model,
		loss_fn=loss_fn,
		train_loader=train_loader,
		val_loader=val_loader,
		period=1
)
