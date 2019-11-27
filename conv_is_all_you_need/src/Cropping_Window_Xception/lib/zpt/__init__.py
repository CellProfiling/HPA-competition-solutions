def _call_plugins(plugins, method_name, state):
    for plugin in plugins:
        state.update(getattr(plugin, method_name)(state))
    return state


# the reason we are using this awkward state dict instead of
# class (which already has the state mechanism) is because
# we want the plugins to be able to write to the state to communicate
# with other plugins (or with the trainer itself)


def train(
    model,
    optimizer,
    train_data,
    loss_fn,
    val_data,
    n_epochs=1,
    online_metrics=[],
    train_metrics=[],
    valid_metrics=[],
    plugins=[],
    device='cuda',
):
    state = {}
    model.to(device)
    for i_epoch in range(n_epochs):
        state.update({
            'i_epoch': i_epoch,
        })
        model.train()
        optimizer.zero_grad()

        for i_batch, batch in enumerate(train_data):
            state.update({
                'i_batch': i_batch,
                'batch': batch,
            })
            x, y = batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            state.update({
                'loss': loss,
            })
            loss.backward()
            optimizer.step()

            state.update(_call_plugins(plugins, 'on_batch_trained', state))

        state.update(_call_plugins(plugins, 'on_epoch_trained', state))

        model.eval()

        for batch in val_data:
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            self.optimizer.step()
            return loss.item()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
