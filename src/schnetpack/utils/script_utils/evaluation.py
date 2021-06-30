import os
import csv


__all__ = ["evaluate", "evaluate_dataset"]


def evaluate(
    args,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    metrics,
    custom_header=None,
):

    header = []
    results = {}
    batch = {}
    results["energy"]=[]
    results["forces"]=[]
    batch["energy"]=[]
    batch["forces"]=[]
    loaders = dict(train=train_loader, validation=val_loader, test=test_loader)
    for datasplit in args.split:
        header += ["{} MAE".format(datasplit), "{} RMSE".format(datasplit)]
        derivative = model.output_modules[0].derivative
        if derivative is not None:
            header += [
                "{} MAE ({})".format(datasplit, derivative),
                "{} RMSE ({})".format(datasplit, derivative),
            ]
        r,b = evaluate_dataset(metrics, model, loaders[datasplit], device)
        
        #for i in range(len(r)):
        #    for j in r[i].keys():
        #        results[i].append(r[i][j].detach().cpu().numpy())
        #        batch[i].append(b[i][j].detach().cpu().numpy())

    if custom_header:
        header = custom_header
    import numpy as np
    #np.savez("prediction.npz",r)#.detach().cpu().numpy())
    #np.savez("reference.npz",b)#.detach().cpu().numpy())
    #for i in results:    
    #    print(i)
    #    np.savez("prediction_%s.npz"%i,results[i])#.detach().cpu().numpy())
    #    np.savez("reference_%s.npz"%i,batch[i])#.detach().cpu().numpy())
    results = [metric.aggregate() for metric in metrics]
    eval_file = os.path.join(args.modelpath, "evaluation.txt")
    with open(eval_file, "w") as file:
        wr = csv.writer(file)
        wr.writerow(header)
        wr.writerow(results)


def evaluate_dataset(metrics, model, loader, device):
    model.eval()

    for metric in metrics:
        metric.reset()

    batches = []
    results={}
    for batch in loader:
        batches.append({k: v for k, v in batch.items()})
        batch = {k: v.to(device) for k,v in batch.items()}
        result = model(batch)
        for i in result.keys():
            if i in results:
                results[i].append(result[i].detach().cpu().numpy())
            else:
                results[i]=[]
                results[i].append(result[i].detach().cpu().numpy())
        for metric in metrics:
            metric.add_batch(batch, result)
    return results,batches
