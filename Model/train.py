

if __name__ == '__main__':
  loss_fn = torch.nn.BCELoss(reduction='mean')
  net = EdgeConvNet()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)


  
  # check if a GPU is available. Otherwise run on CPU
  device = 'cpu'
  args_cuda = torch.cuda.is_available()
  if args_cuda: device = "cuda:0"
  print('device : ',device)
  net.to(device)
  
  losses = []
  nepochs = 1
  net.train()
  for i in range(nepochs):
    print(f'EPOCH {i}')
    running_loss = 0.
    for batchnum, batch in enumerate(train_loader):
      optimizer.zero_grad()
      batch.to(device)
      pred = net(batch)
      loss = loss_fn(pred, batch.edge_label.reshape(len(batch.edge_label), 1).float())
      loss.backward()
      optimizer.step()
      theloss = loss.item()
      running_loss += theloss
      if not batchnum % 10: print(f'{batchnum}')
      if not batchnum % 100 and batchnum > 0:
        print(f'\n(Batch {batchnum}) Loss: {running_loss / 100.}')
        running_loss = 0.
      losses.append(theloss)
