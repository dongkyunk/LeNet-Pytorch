from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


import torch
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate(val_loader, model, criterion, epoch):
    if DEVICE == 'cuda':
        model.cuda()
    model.eval()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            inp = inp.cuda()
            target = target.cuda()
            output = model(inp)


            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, 100.0*losses.avg, 100.0*nme,
                                failure_008_rate, failure_010_rate)
    print(msg)

    return nme, predictions