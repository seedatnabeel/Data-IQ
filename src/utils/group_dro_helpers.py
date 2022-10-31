# third party
import torch
import torch.nn as nn


def train_loop(net, criterion, EPOCHS, train_loader, optimizer, device, subclass=True):
    """
    > This function takes in a model, a loss function (criterion), the number of epochs, a dataloader, an optimizer,
    and a device. It then trains the model for the number of epochs specified, and returns the trained
    model

    Args:
      net: the network we're training
      criterion: the loss function
      EPOCHS: number of epochs to train for
      train_loader: the training data
      optimizer: the optimizer we're using to train the network
      device: the device we're using to train the model.
      subclass: whether or not we're using a subclass. Defaults to True

    Returns:
      The trained model
    """
    for e in range(1, EPOCHS + 1):
        net.train()
        epoch_loss = 0
        epoch_acc = 0

        # if a subclass is present how we evaluate the loss differently
        if subclass:
            for X_batch, y_batch, y_subclass in train_loader:
                X_batch, y_batch, y_subclass = (
                    X_batch.to(device),
                    y_batch.to(device),
                    y_subclass.to(device),
                )
                optimizer.zero_grad()
                sf = nn.LogSoftmax()
                y_pred = net(X_batch)

                _, predicted = torch.max(y_pred.data, 1)

                y_batch = y_batch.to(torch.int64)
                co = criterion(sf(y_pred), y_batch, y_subclass)
                loss, (losses, corrects), _ = co

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += (predicted == y_batch).sum().item() / len(y_batch)
        else:
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                sf = nn.LogSoftmax()
                y_pred = net(X_batch)

                _, predicted = torch.max(y_pred.data, 1)

                y_batch = y_batch.to(torch.int64)

                loss = criterion(sf(y_pred), y_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += (predicted == y_batch).sum().item() / len(y_batch)

        print(
            f"Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}",
        )

    return net


def evaluate_model(net_test, X_test, y_test, easy_test, ambig_test, hard_test):
    """
    It takes a trained model, a test set, and the indices of the easy, ambiguous, and hard test
    examples, and returns the performance of the model on each of these subgroups

    Args:
      net_test: the model you want to evaluate
      X_test: the test data
      y_test: the true labels for the test set
      easy_test: indices of easy test examples
      ambig_test: the indices of the ambiguous test set
      hard_test: the indices of the hard test set

    Returns:
      a dictionary with the overall accuracy, the accuracy on the rest of the data, and the accuracy on
    the ambiguous data.
    """
    # third party
    import numpy as np
    from sklearn.metrics import accuracy_score

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_test.eval()
    with torch.no_grad():
        X_batch = torch.FloatTensor(X_test)
        X_batch = X_batch.to(device)
        y_test_pred = net_test(X_batch)

    threshold = 0.5
    preds = y_test_pred.data[:, 1].cpu().numpy()
    overall = accuracy_score(preds > threshold, y_test)

    partition = np.hstack([easy_test])
    with torch.no_grad():
        X_batch = torch.FloatTensor(X_test[partition, :])
        X_batch = X_batch.to(device)
        y_test_pred = net_test(X_batch)

    threshold = 0.5
    preds = y_test_pred.data[:, 1].cpu().numpy()
    try:
        rest = accuracy_score(preds > threshold, y_test[partition])
    except BaseException:
        rest = accuracy_score(preds > threshold, y_test.to_numpy()[partition])

    partition = ambig_test
    with torch.no_grad():
        X_batch = torch.FloatTensor(X_test[partition, :])
        X_batch = X_batch.to(device)
        y_test_pred = net_test(X_batch)

    threshold = 0.5
    preds = y_test_pred.data[:, 1].cpu().numpy()
    try:
        ambig = accuracy_score(preds > threshold, y_test[partition])
    except BaseException:
        ambig = accuracy_score(preds > threshold, y_test.to_numpy()[partition])

    res = {}
    res["overall"] = overall
    res["rest"] = rest
    res["ambig"] = ambig
    return res


# def evaluate_model2(net_test, X_test, y_test, easy_test, incons_test, hard_test):
#     from sklearn.metrics import accuracy_score

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     net_test.eval()
#     with torch.no_grad():
#         X_batch = torch.FloatTensor(X_test)
#         X_batch = X_batch.to(device)
#         y_test_pred = net_test(X_batch)

#     threshold = 0.5
#     preds = y_test_pred.data[:, 1].cpu().numpy()
#     print("Overall score = ", accuracy_score(preds > threshold, y_test))
#     overall = accuracy_score(preds > threshold, y_test)

#     partition = easy_test
#     with torch.no_grad():
#         X_batch = torch.FloatTensor(X_test[partition, :])
#         X_batch = X_batch.to(device)
#         y_test_pred = net_test(X_batch)

#     threshold = 0.5
#     preds = y_test_pred.data[:, 1].cpu().numpy()
#     print("EASY = ", accuracy_score(preds > threshold, y_test[partition]))
#     easy = accuracy_score(preds > threshold, y_test[partition])

#     partition = incons_test
#     with torch.no_grad():
#         X_batch = torch.FloatTensor(X_test[partition, :])
#         X_batch = X_batch.to(device)
#         y_test_pred = net_test(X_batch)

#     threshold = 0.5
#     preds = y_test_pred.data[:, 1].cpu().numpy()
#     print("INCONS = ", accuracy_score(preds > threshold, y_test[partition]))
#     incons = accuracy_score(preds > threshold, y_test[partition])

#     partition = hard_test
#     with torch.no_grad():
#         X_batch = torch.FloatTensor(X_test[partition, :])
#         X_batch = X_batch.to(device)
#         y_test_pred = net_test(X_batch)

#     threshold = 0.5
#     preds = y_test_pred.data[:, 1].cpu().numpy()
#     print("HARD = ", accuracy_score(preds > threshold, y_test[partition]))
#     hard = accuracy_score(preds > threshold, y_test[partition])

#     res = {}
#     res["overall"] = overall
#     res["easy"] = easy
#     res["incons"] = incons
#     res["hard"] = hard
#     return res
