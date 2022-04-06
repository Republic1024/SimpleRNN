import datetime  # <1>
import torch

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))


def accuracy(model, val_loader):
	correct = 0
	total = 0
	with torch.no_grad():
		for imgs, labels in val_loader:
			imgs = imgs.to(device=device)
			labels = labels.to(device=device)
			outputs = model(imgs)
			_, predicted = torch.max(outputs, dim=1)  # <1>
			total += labels.shape[0]
			correct += int((predicted == labels).sum())

	# print("Accuracy: {:.2f}".format(correct / total))
	return correct / total


def validate(model, train_loader, val_loader):
	accdict = {}
	for name, loader in [("train", train_loader), ("val", val_loader)]:
		correct = 0
		total = 0

		with torch.no_grad():
			for imgs, labels in loader:
				imgs = imgs.to(device=device)
				labels = labels.to(device=device)
				outputs = model(imgs)
				_, predicted = torch.max(outputs, dim=1)  # <1>
				total += labels.shape[0]
				correct += int((predicted == labels).sum())

		print("Accuracy {}: {:.2f}".format(name, correct / total))
		accdict[name] = correct / total
	return accdict


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
	for epoch in range(1, n_epochs + 1):  # <2>
		loss_train = 0.0
		for imgs, labels in train_loader:  # <3>
			outputs = model(imgs)  # <4>
			loss = loss_fn(outputs, labels)  # <5>
			optimizer.zero_grad()  # <6>
			loss.backward()  # <7>
			optimizer.step()  # <8>
			loss_train += loss.item()  # <9>
		# print(imgs.shape,labels.shape)

		if epoch == 1 or epoch % 10 == 0:
			print('{} Epoch {}, Training loss {}'.format(
					datetime.datetime.now(), epoch,
					loss_train / len(train_loader)))  # <10>


def training_loop_with_acc(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, period=1):
	acc_list = []
	for epoch in range(1, n_epochs + 1):  # <2>
		loss_train = 0.0
		for imgs, labels in train_loader:  # <3>
			outputs = model(imgs)  # <4>
			loss = loss_fn(outputs, labels)  # <5>
			optimizer.zero_grad()  # <6>
			loss.backward()  # <7>
			optimizer.step()  # <8>
			loss_train += loss.item()  # <9>

		if epoch == 1 or epoch % period == 0:
			acc = accuracy(model, val_loader=val_loader)
			print('{} Epoch {}, Training loss {}, Accuracy: {}'.format(
					datetime.datetime.now(), epoch,
					loss_train / len(train_loader), acc))  # <10>
			acc_list.append(acc)

	return acc_list
