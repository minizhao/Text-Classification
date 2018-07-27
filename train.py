from Models import Encoder
from data_loader import cropus
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import torch.optim as optim
import os
from visdom import Visdom


def get_dev_loss(model,mycropus,dev_data,criterion):
    #计算开发测试节集损失
    dev_loss_list=[]
    dev_acc_list=[]
    for batch_data in mycropus.batch_iterator(dev_data):
        inp,pos,tag=[x[0] for x in batch_data],[x[1] for x in batch_data],[x[2] for x in batch_data]
        inp=Variable(torch.from_numpy(np.array(inp)).cuda())
        pos=Variable(torch.from_numpy(np.array(pos)).cuda())
        tag=Variable(torch.LongTensor(torch.from_numpy(np.array(tag)))).cuda()

        preds= model(inp, pos)

        loss = criterion(preds,tag)
        _,pred_idx=torch.max(preds,1)
        dev_acc_list.append((sum(pred_idx.cpu().data.numpy()==tag.cpu().data.numpy())*1./tag.size(0)))
        dev_loss_list.append(loss.cpu().data.numpy())
    print("mean_dev_loss:{},mean_dev_acc:{}".format(np.mean(dev_loss_list),np.mean(dev_acc_list)))
    return np.mean(dev_acc_list),np.mean(dev_loss_list)


def train():

    viz = Visdom()
    line = viz.line(np.arange(2))

    mycropus=cropus()
    n_class=mycropus.n_class

    if os.path.isfile('save/model.pt'):
        model=torch.load('save/model.pt')
    else:
        model=Encoder(n_src_vocab=len(mycropus.token2idx),n_max_seq=mycropus.max_len).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer=optim.Adam(model.parameters(),lr=0.0005)

    train_loss_p=[]
    train_acc_p=[]
    dev_loss_p=[]
    dev_acc_p=[]

    step_p=[]

    for epoch in range(100):
        step=0
        tr_loss_list=[]
        tr_acc_list=[]
        best_dev_acc=0.3
        for batch_data in mycropus.batch_iterator(mycropus.train_data):
            inp,pos,tag=[x[0] for x in batch_data],[x[1] for x in batch_data],[x[2] for x in batch_data]
            inp=Variable(torch.from_numpy(np.array(inp)).cuda())
            pos=Variable(torch.from_numpy(np.array(pos)).cuda())
            tag=Variable(torch.LongTensor(torch.from_numpy(np.array(tag)))).cuda()

            preds= model(inp, pos)
            loss = criterion(preds,tag)

            _,pred_idx=torch.max(preds,1)
            tr_acc_list.append((sum(pred_idx.cpu().data.numpy()==tag.cpu().data.numpy())*1./tag.size(0)))
            tr_loss_list.append(loss.cpu().data.numpy())
            optimizer.zero_grad()
            loss.backward()
            # 剪裁参数梯度
            nn.utils.clip_grad_norm(model.parameters(), 1, norm_type=2)
            optimizer.step()
            step=step+1

            if step%100==0:
                print("epoch:{},step:{},mean_loss:{},mean_acc:{}".format(epoch,step,np.mean(tr_loss_list),np.mean(tr_acc_list)))
                dev_acc,dev_loss=get_dev_loss(model,mycropus,mycropus.dev_data,criterion)
                if best_dev_acc<dev_acc:
                    torch.save(model,'save/model.pt')
                    best_dev_acc=dev_acc
                print("-----------")

                train_loss_p.append(np.mean(tr_loss_list))
                train_acc_p.append(np.mean(tr_acc_list))
                dev_loss_p.append(np.mean(dev_loss))
                dev_acc_p.append(np.mean(dev_acc))

                step_p.append(step+epoch*mycropus.nums_batch)
                viz.line(
                     X=np.column_stack((np.array(step_p), np.array(step_p),np.array(step_p), np.array(step_p))),
                     Y=np.column_stack((np.array(train_loss_p),np.array(train_acc_p),np.array(dev_loss_p), np.array(dev_acc_p))),
                     win=line,
                    opts=dict(legend=["Train_mean_loss", "Train_acc","Eval_mean_loss", "Eval_acc"]))

                tr_loss_list=[]
                tr_acc_list=[]


def preds():
    mycropus=cropus()
    if os.path.isfile('save/model.pt'):
        model=torch.load('save/model.pt')
    else:
        print("no model file")
        sys.exit(0)
    # 开始生产测试结果，
    print("test data  size :{},batch nums:{}".format(len(mycropus.test_data),len(mycropus.test_data)/mycropus.batch_size))
    with open("save/results.csv",'w') as f:
        step=0
        for batch_data in mycropus.batch_iterator(mycropus.test_data):
            # 对测试集来说，最后一个文本id
            inp,pos,id=[x[0] for x in batch_data],[x[1] for x in batch_data],[x[2] for x in batch_data]
            inp=Variable(torch.from_numpy(np.array(inp)).cuda())
            pos=Variable(torch.from_numpy(np.array(pos)).cuda())
            preds= model(inp, pos)
            _,pred_idx=torch.max(preds,1)
            output=pred_idx.cpu().data.numpy()
            step=step+1
            if step%100==0:
                print("step:{}".format(step))
            for id,label in zip(id,output):
                f.write(str(id)+","+str(label+1)+"\n")



if __name__ == '__main__':
    preds()
