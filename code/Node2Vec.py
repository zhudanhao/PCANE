
# coding: utf-8

# In[1]:


import random
import codecs
import tensorflow as tf
import numpy as np
import evaluation
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
# In[2]:

'''
def weight_choice(weight):
    t = random.randint(0, sum(weight) - 1)
    for i, val in enumerate(weight):
        t -= val
        if t < 0:
            return i
'''
def weight_choice(probabilities):  
    some_list = range(len(probabilities))
    sum_p = sum(probabilities)
    probabilities = [p/sum_p for p in probabilities]
    x = random.uniform(0,1)  
    cumulative_probability = 0.0  
    for item, item_probability in zip(some_list, probabilities):  
        cumulative_probability += item_probability  
        if x < cumulative_probability:break  
    return item 
def n2v_walks(links,walk_length = 10 ,p = 1,q = 0.25):
        #links = self.links
        nodes_map = dict()
        for a,b in links:
            if a not in nodes_map:
                nodes_map[a] = set()
            nodes_map[a].add(b)

        for a in nodes_map:
            nodes_map[a] = list(nodes_map[a])

        paths = []
        for num_walks in range(80):
            for a in nodes_map:
                #不是随机采样，而是对于每一个节点走
                path = []
                now = a
                path.append(now)
                index = random.randint(0,len(nodes_map[now])-1)
                path.append(nodes_map[now][index])
                last = a
                now = nodes_map[now][index]

                for step in range(walk_length-2):       
                    if now not in nodes_map:
                        break
                    lst = nodes_map[now]
                    weight = []
                    for l in lst:
                        if l==last:
                            weight.append(1/p)
                            continue
                        if l in nodes_map[last]:
                            weight.append(1)
                            continue
                        weight.append(1/q)
                    last = now
                    now = lst[weight_choice(weight)]                                           
                    path.append(now)
                paths.append(path)
            

        return paths



def re_index(links,nodes_dic=None):
    if nodes_dic is None:
        nodes_dic = dict()
    i = 0
    xs,ys = [],[]
    for a,b in links:
        if a not in nodes_dic:
            nodes_dic[a] = i
            i+=1
        if b not in nodes_dic:
            nodes_dic[b] = i
            i+=1    
        xs.append(nodes_dic[a])
        ys.append(nodes_dic[b])
        
    return np.array(xs),np.array(ys),nodes_dic
        

def generate_test_data(nodes_dic,all_xs_data,all_ys_data,test_xs_data,test_ys_data,neg_ratio=1.0):
    test_data = []
    for x,y in zip(test_xs_data,test_ys_data):
        test_data.append([x,y,1])
    #mix with negative data
    neg_size = int(len(test_xs_data)*neg_ratio)
    random.seed(2002)
    
    search_set = set()
    for x,y in zip(all_xs_data,all_ys_data):
        search_set.add(str(x)+' '+str(y))
    while len(test_data) < neg_size+len(test_xs_data):
        x = random.randint(0,len(nodes_dic)-1)
        y = random.randint(0,len(nodes_dic)-1)
        if str(x)+' '+str(y) in search_set or str(y)+' '+str(x) in search_set:
            continue
        test_data.append([x,y,0])
        search_set.add(str(x)+' '+str(y))
    return test_data

                
def main(args):
    
    #train_link_file = '../../baseline_20180202/data/ego/train_0.9.edgelist'
    #all_link_file = '../../baseline_20180202/data/ego/ego_shuffle.edgelist'
    train_link_file = args.train_link_file
    all_link_file = args.all_link_file
    training_percent = args.training_percent
    dataset = args.dataset
    n2v_p = args.p
    n2v_q = args.q
    print('p',n2v_p,'q',n2v_q)
    
    for s in [train_link_file,all_link_file,training_percent,dataset]:
        print(s)
    
    train_links_str = [] 
    with codecs.open(train_link_file) as f:
        for line in f:
            train_links_str.append(line)
    all_links_str = [] 
    with codecs.open(all_link_file) as f:
        for line in f:
            all_links_str.append(line)
    print(len(train_links_str))


    train_links, test_links, all_links = [],[],[]
    test_links_str = set(all_links_str) - set(train_links_str)
    #将所有的link进行训练，用作linkprediction task
    #train_links_str = all_links_str
    #
    for line in train_links_str:
        a,b = line.split()
        train_links.append([int(a),int(b)]) 
    
    for line in test_links_str:
        a,b = line.split()
        test_links.append([int(a),int(b)]) 
    for line in all_links_str:
        a,b = line.split()
        all_links.append([int(a),int(b)]) 
        
        
    all_xs_data,all_ys_data,nodes_dic = re_index(all_links)
    train_xs_data,train_ys_data,_ = re_index(train_links,nodes_dic)
    test_xs_data,test_ys_data,_ = re_index(test_links,nodes_dic)
    num_nodes = len(nodes_dic)
    print(num_nodes)
    #训练数据变成双向
    temp_a,temp_b = [],[]
    for a,b in zip(train_xs_data,train_ys_data):
        temp_a.append(a)
        temp_b.append(b)
        temp_a.append(b)
        temp_b.append(a)
    train_xs_data,train_ys_data = temp_a,temp_b
    
    
    test_data = generate_test_data(nodes_dic,all_xs_data,all_ys_data,test_xs_data,test_ys_data,neg_ratio=1.0)

    sents = n2v_walks(zip(train_xs_data,train_ys_data),p=n2v_p,q=n2v_q)


    random.shuffle(sents)
    train_xs_data,train_ys_data = [], []
    for sent in sents:
        for i in range(len(sent)):
            for j in range(len(sent)):
                if i!=j and (i-j<=5 and i-j>=-5):
                    a,b = sent[i],sent[j]
                    train_xs_data.append(a)
                    train_ys_data.append(b)



    train_xs_data = np.array(train_xs_data)
    train_ys_data = np.array(train_ys_data)

    #Computational Graph Definition
    tf.reset_default_graph()#remove this if not ipython notebook
    batch_size = 1024
    embedding_size = 128 # Dimension of the embedding vector.
    num_sampled = 64 # Number of negative examples to sample.


    node_embeddings_in = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
    node_embeddings_out = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0,1.0))

    #Fixedones
    biases=tf.Variable(tf.zeros([num_nodes]))

    train_xs =  tf.placeholder(tf.int32, shape=[None],name="xs")
    train_ys =  tf.placeholder(tf.int32, shape=[None,1],name="xs")

    xs_emb = tf.nn.embedding_lookup(node_embeddings_in, train_xs)


    loss_node2vec = tf.reduce_mean(tf.nn.sampled_softmax_loss(node_embeddings_out,
                                                              biases,train_ys,xs_emb, num_sampled, num_nodes))
    update_loss = tf.train.AdamOptimizer().minimize(loss_node2vec)
    init = tf.initialize_all_variables()


    loss_rec = []
    batch_num = len(train_xs_data) // batch_size
    
    max_roc = -1
    rocs = []
    stop = False
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(5):
            if stop:
                break
            for i in range(batch_num):            
                data_xs =train_xs_data[i*batch_size:(i+1)*batch_size]
                data_ys = train_ys_data[i*batch_size:(i+1)*batch_size].reshape([-1,1])
                feed_dict={
                           train_xs:data_xs,
                           train_ys:data_ys,
                          }        
                _,loss_value=sess.run([update_loss,loss_node2vec], feed_dict)
                if i%(int(batch_num/10)) == 0:        
                    Embeddings = sess.run(node_embeddings_in)
                    roc = evaluation.evaluate_ROC(test_data, Embeddings)
                    rocs.append(roc)
                    print('epoch',epoch,'batch_num',i,'roc:',roc,'loss:',loss_value)
                    if roc>max_roc:
                        #np.save('n2v_emb_'+dataset+'_'+training_percent,[nodes_dic,Embeddings])
                        max_roc = roc
                    if len(rocs)>5 and rocs[-1]< min(rocs[-6:-1]):
                        print('stop for no progress')
                        stop = True
                        break
        #np.save('n2v_trainingrocs_'+dataset+'_'+training_percent,rocs)
        print('n2v,',dataset,training_percent,'max roc:',max_roc,'p',n2v_p,'q',n2v_q)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_link_file', default='', type=str,
                        help='training link file')
    parser.add_argument('--all_link_file', default='', type=str,
                        help='all link file')
    parser.add_argument('--training_percent', default='0.1', type=str,
                        help='training percentage, not for compute, just for save file')
    parser.add_argument('--dataset', default='', type=str,
                        help='data name, not for compute, just for save file')
    parser.add_argument('--p', default=1, type=float,
                        help='node2vec p')
    parser.add_argument('--q', default=0.25, type=float,
                        help='node2vec q')
    
    
    
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    args = parser()
    main(args)        
    
