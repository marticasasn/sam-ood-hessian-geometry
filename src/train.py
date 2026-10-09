import torch


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_single_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    use_sam = hasattr(optimizer, "first_step")

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        if use_sam:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            criterion(model(images), labels).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(train_loader), 100.0 * correct / total
