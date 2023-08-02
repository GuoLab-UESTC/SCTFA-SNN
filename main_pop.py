from parameters import *
import net_pop
import copy

import time

def train(cs, sctfa=False, batch_size=init_batch_size):
    train_dataset = MyDataset(datasetPath=datasetPath, sampleFile=datapath[1000:], Train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    test_dataset = MyDataset(datasetPath=datasetPath, sampleFile=datapath[:1000], Train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=12)

    acc_record = list([])
    loss_train_record = list([])
    loss_test_record = list([])

    mynn = net_pop.snn(sctfa)
    mynn.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mynn.parameters(), lr=learning_rate)
    exprlr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    max_correct = 0
    ithrehsh = 0

    for epoch in range(num_epochs):
        print(name, cs, sctfa)
        running_loss = 0
        train_avg_loss = 0
        mynn.train()
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            mynn.zero_grad()
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = mynn(images)
            loss = criterion(outputs, F.one_hot(labels, 10).float())
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            ithrehsh += 1
            if (i + 1) % 25 == 0:
                train_avg_loss += running_loss
                running_loss = 0

        print('\n')
        loss_train_record.append(train_avg_loss)
        print('Loss_train1 :', train_avg_loss)

        exprlr.step()

        correct = 0
        test_avg_loss = 0
        total = 1000
        mynn.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                outputs = mynn(inputs)
                labels_ = torch.zeros(inputs.size(0), 10).scatter_(1, targets.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                test_avg_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += float(predicted.eq(targets.to(device)).sum().item())
        loss_test_record.append(test_avg_loss)
        print('Loss_test  :', test_avg_loss)
        print('Test Accuracy  : %.3f' % (100 * correct / total))
        print('Iters:', epoch + 1, '\n')
        print(time.time() - start_time)

        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)

        if max_correct <= correct:
            max_correct = correct
            savemodel = copy.deepcopy(mynn.state_dict())

        if (epoch + 1) % 25 == 0:
            print('Saving..', '\n\n\n')
            state = {
                'net': savemodel,
                'acc_record': acc_record,
                'max_acc': max_correct,
                'loss_train_record': loss_train_record,
                'loss_test_record': loss_test_record,
                'decay': decay,
                'gamma': gamma,
                'thresh': thresh,
                'learning_rate': learning_rate,
                'init_batch_size': init_batch_size
            }
            torch.save(state, savemodel_path + '/checkpoint' + str(cs) + '.t7')



if __name__ == '__main__':
    for cs in range(10):
        train(cs, sctfa=True)
