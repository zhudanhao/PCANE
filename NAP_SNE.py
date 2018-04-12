
# coding: utf-8


import random
import codecs
import tensorflow as tf
import numpy as np
import evaluation
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
import math
# In[2]:
# In[5]:

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


# In[6]:


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



#将属性按类型分解
def extract_attr(att_dic):
        att_type_dic = dict()
        for item in att_dic:
            c,v = item.split('_')
            if c not in att_type_dic:
                att_type_dic[c] = set()
            att_type_dic[c].add(v)
        att_voc_dic = dict()
        for att_type in att_type_dic:
            val_list = list(att_type_dic[att_type])
            voc_dic = dict()
            for i in range(len(val_list)):
                voc_dic[val_list[i]] = i
            att_voc_dic[att_type] = voc_dic
        return att_voc_dic 



# In[27]:


def build_batch_attr_data(xarr,node_att_dic,attr_num):
    att_arr = np.zeros([len(xarr),attr_num],dtype=np.float32)
    for i in range(len(xarr)):
        node_id = xarr[i]
        for att_id in node_att_dic[node_id]:
            att_arr[i][att_id] = 1
    return att_arr


# In[64]:


def build_batch_attr_data(xarr,node_att_dic,attr_num):
    att_arr = np.zeros([len(xarr),attr_num+1],dtype=np.float32)
    for i in range(len(xarr)):
        node_id = xarr[i]
        if node_id in node_att_dic:
            for att_id in node_att_dic[node_id]:
                att_arr[i][att_id] = 1
        att_arr[i][-1] = 1 
    return att_arr

def build_attr_labels(xarr,yarr,attr_arr,node_attr_dic,attr_voc_dic,attr_dic):
    attr_rev_dic = {}
    for a in attr_dic:
        attr_rev_dic[attr_dic[a]] = a
    data_attr_dict = dict()
    for attr_type in attr_voc_dic:
        data_attr_dict[attr_type] = []
    for i in range(len(attr_arr)):
        if yarr[i][0] in node_attr_dic:
            for attr_id in node_attr_dic[yarr[i][0]]:
                attr_str = attr_rev_dic[attr_id]
                attr_type,attr_value= attr_str.split('_')
                data_attr_dict[attr_type].append([xarr[i],attr_arr[i],attr_voc_dic[attr_type][attr_value]])
    return data_attr_dict


# In[65]:

# Add ops to save and restore all the variables.

# Optimizer.


# In[66]:

'''
def concat_full_embeddings(embeddings,attr_embddings,node_att_dic):
    node_attr_embeddings = np.zeros([embeddings.shape[0],attr_embddings.shape[1]])
    for i in range(len(embeddings)):
        att_ids = node_att_dic[i]
        for att_id in att_ids:
            node_attr_embeddings[i] += attr_embddings[att_id]
    return np.concatenate([embeddings,node_attr_embeddings],1),node_attr_embeddings
'''
def concat_full_embeddings(embeddings,attr_embddings,node_att_dic):
    node_attr_embeddings = np.zeros([embeddings.shape[0],attr_embddings.shape[1]])
    for i in range(len(embeddings)):
        if i in node_att_dic:
            att_ids = node_att_dic[i]
            for att_id in att_ids:
                node_attr_embeddings[i] += attr_embddings[att_id]
        node_attr_embeddings[i] += attr_embddings[len(attr_embddings)-1]
    return np.concatenate([embeddings,node_attr_embeddings],1),node_attr_embeddings

# In[79]:


def fit_attr(data_xs,data_ys,data_attr,nodes_attr_int_map,attr_voc_dic,attr_map,
             train_xs,train_ys,attr_in,train_labels_attrs,loss_attr,optimizer_attr,sess):
        data_attr_dic = build_attr_labels(data_xs,data_ys,data_attr,nodes_attr_int_map,attr_voc_dic,attr_map)
        for attr_type in data_attr_dic:
            xid,xattr,attr_label = [],[],[]
            if len(data_attr_dic[attr_type]) ==0:
                continue
            for a,b,c in data_attr_dic[attr_type]:
                xid.append(a)
                xattr.append(b)
                attr_label.append(c)
            xid, xattr,attr_label = np.array(xid),np.array(xattr),np.array(attr_label).reshape([-1,1])
            #print(xattr.shape)
            feed_dict = {train_xs: xid, attr_in: xattr,
                     train_labels_attrs[attr_type]: attr_label}
            loss, opt = sess.run((loss_attr[attr_type], optimizer_attr[attr_type]), feed_dict=feed_dict)
        return loss


# In[ ]:

def main(args):

    train_link_file = args.train_link_file
    all_link_file = args.all_link_file
    attr_file = args.attr_file
    training_percent = args.training_percent
    dataset = args.dataset
    attr_alpha = args.attr_alpha
    n2v_p = args.p
    n2v_q = args.q
    print('attr_alpha',attr_alpha,'p',n2v_p,'q',n2v_q)
    
    for s in [train_link_file,all_link_file,attr_file,training_percent,dataset]:
        print(s)

    train_links_str = [] 
    with codecs.open(train_link_file) as f:
        for line in f:
            train_links_str.append(line)
    all_links_str = [] 
    with codecs.open(all_link_file) as f:
        for line in f:
            all_links_str.append(line)



    # In[7]:


    attr_map = {}
    node_attr_map = {}
    i = 0
    with codecs.open(attr_file) as f:
        for line in f:
            arr = line.split()
            if len(arr)>1:
                for att in arr[1:]:
                    if att not in attr_map:
                        attr_map[att] = i
                        i += 1
            node_attr_map[arr[0]] = [] 
            for att in arr[1:]:
                node_attr_map[arr[0]].append(attr_map[att])


    # In[8]:


    test_links_str = set(all_links_str) - set(train_links_str)
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


    # In[9]:


    all_xs_data,all_ys_data,nodes_dic = re_index(all_links)
    train_xs_data,train_ys_data,_ = re_index(train_links,nodes_dic)
    test_xs_data,test_ys_data,_ = re_index(test_links,nodes_dic)
    num_nodes = len(nodes_dic)


    # In[11]:


    ##属性建造为一个dic，key为属性的int key，值为属性列表，int，均为编码过后的
    nodes_attr_int_map = {}
    no_attr_num = 0 
    has_attr_num = 0
    for key in nodes_dic:
        if str(key) not in node_attr_map:
            no_attr_num+=1
            continue
        nodes_attr_int_map[nodes_dic[key]] = node_attr_map[str(key)]
        has_attr_num += 1
    print('no attrs nodes:',no_attr_num ,'with attrs nodes:',has_attr_num)
    
    #nodes_attr_int_map  key: 正式的node标志， values：正式的属性列表


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


    # In[16]:



    random.shuffle(sents)
    train_xs_data,train_ys_data = [], []
    for sent in sents:
        for i in range(len(sent)):
            for j in range(len(sent)):
                if i!=j and (i-j<=5 and i-j>=-5):
                    a,b = sent[i],sent[j]
                    train_xs_data.append(a)
                    train_ys_data.append(b)


    # In[17]:


    train_xs_data = np.array(train_xs_data)
    train_ys_data = np.array(train_ys_data)
    len(train_xs_data)
    att_voc_dic = extract_attr(attr_map)


    #Computational Graph Definition
    tf.reset_default_graph()#remove this if not ipython notebook
    batch_size = 1024
    embedding_size = 100 # Dimension of the embedding vector.
    attr_emb_size = 28
    num_sampled = 64 # Number of negative examples to sample.
    attr_num = len(attr_map)

    node_embeddings_in = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
    node_embedding_attr_in = tf.Variable(tf.random_uniform([attr_num+1, attr_emb_size], -1.0, 1.0))

    node_embeddings_out = tf.Variable(tf.random_uniform([num_nodes, embedding_size+attr_emb_size], -1.0,1.0))



    #Fixedones
    biases=tf.Variable(tf.zeros([num_nodes]))

    train_xs =  tf.placeholder(tf.int32, shape=[None],name="xs")
    train_ys =  tf.placeholder(tf.int32, shape=[None,1],name="xs")
    attr_in = tf.placeholder(tf.float32, shape=[None,attr_num+1],name="attr_in")
    train_labels_attrs = dict()
    for attr_type in att_voc_dic:
        train_labels_attrs[attr_type] = tf.placeholder(tf.int32, shape=[None, 1],name='attr_type_'+attr_type)
    all_weights = {}
    for att_type in att_voc_dic:
         all_weights[att_type+'_w'] = tf.Variable(
             tf.truncated_normal([len(att_voc_dic[att_type]), embedding_size + attr_emb_size], 
                                 stddev=1.0 / math.sqrt(embedding_size + attr_emb_size)))   
         all_weights[att_type+'_b'] = tf.Variable(tf.zeros([len(att_voc_dic[att_type])]))

    xs_emb = tf.nn.embedding_lookup(node_embeddings_in, train_xs)
    attr_in_emb_sum = tf.matmul(attr_in, node_embedding_attr_in) 

    embed_layer = tf.concat([xs_emb, attr_in_emb_sum],1) 
    #weights = tf.Variable(tf.random_uniform([ embedding_size,num_nodes], -1.0,1.0))

    #logits = tf.matmul(xs_emb,weights) + biases

    #ys_one_hot = tf.one_hot(train_ys,num_nodes)

    #loss_node2vec = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=ys_one_hot,logits=logits))/batch_size


    loss_node2vec = tf.reduce_mean(tf.nn.sampled_softmax_loss(node_embeddings_out,
                                                              biases,train_ys,embed_layer, num_sampled, num_nodes))
    update_loss = tf.train.AdamOptimizer().minimize(loss_node2vec)
    #update_loss = tf.train.GradientDescentOptimizer(0.0001).minimize(loss_node2vec)

    #update_loss = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss_node2vec)
    # Initializing the variables

    
    loss_attr = dict()
    optimizer_attr = dict()
    for attr_type in att_voc_dic:
        logits = tf.matmul(embed_layer, tf.transpose(all_weights[attr_type+'_w'])) + all_weights[attr_type+'_b']
        one_hot = tf.one_hot(train_labels_attrs[attr_type],depth=len(att_voc_dic[attr_type]))
        loss_attr[attr_type] = attr_alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=one_hot, logits=logits))
        optimizer_attr[attr_type] = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                                                     epsilon=1e-8).minimize(loss_attr[attr_type])
    init = tf.initialize_all_variables()


    loss_rec = []
    batch_num = len(train_xs_data) // batch_size
    with tf.Session() as sess:
        sess.run(init)
        max_roc = -1
        idrocs,attrrocs,rocs = [],[],[]
        stop = False
        for epoch in range(5):
            if stop:
                break
            for i in range(batch_num):            
                data_xs =train_xs_data[i*batch_size:(i+1)*batch_size]
                data_ys = train_ys_data[i*batch_size:(i+1)*batch_size].reshape([-1,1])
                data_attr = build_batch_attr_data(data_xs,nodes_attr_int_map,attr_num)
                feed_dict={
                           train_xs:data_xs,
                           train_ys:data_ys,
                           attr_in:data_attr
                          }        
                _,loss_value=sess.run([update_loss,loss_node2vec], feed_dict)
                fit_attr(data_xs,data_ys,data_attr,nodes_attr_int_map,att_voc_dic,attr_map,
                         train_xs,train_ys,attr_in,train_labels_attrs,loss_attr,optimizer_attr,sess)
                if i%(int(batch_num/10)) == 0:        
                    Embeddings = sess.run(node_embeddings_in)
                    Embeddings_attr = sess.run(node_embedding_attr_in)
                    Embeddings_all,node_attr_embeddings = concat_full_embeddings(Embeddings,Embeddings_attr,nodes_attr_int_map)
                    roc = evaluation.evaluate_ROC(test_data, Embeddings_all)
                    rocs.append(roc)
                    print('all dim, epoch',epoch,'batch_num',i,'roc:',roc,'loss:',loss_value)
                    '''
                    idroc = evaluation.evaluate_ROC(test_data, Embeddings)
                    idrocs.append(idroc)
                    print('id dim, epoch',epoch,'batch_num',i,'roc:',idroc)
                    attrroc = evaluation.evaluate_ROC(test_data, node_attr_embeddings)
                    attrrocs.append(attrroc)
                    print('attr dim, epoch',epoch,'batch_num',i,'roc:',attrroc)
                    '''
                    if roc>max_roc:
                        #np.save('neap_emb_'+dataset+'_'+training_percent+'alpha'+str(attr_alpha),[nodes_dic,Embeddings_all])
                        max_roc = roc
                    if len(rocs)>5 and rocs[-1]< min(rocs[-6:-1]):
                        print('stop for no progress')
                        stop = True
                        break
        #np.save('neap_trainingrocs_'+dataset+'_'+training_percent,[rocs,idrocs,attrrocs])
        print('neap,',dataset,training_percent,'attr_alpha',attr_alpha,'max roc:',max_roc,'p',n2v_p,'q',n2v_q)
  




def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_link_file', default='', type=str,
                        help='training link file')
    parser.add_argument('--all_link_file', default='', type=str,
                        help='all link file')
    parser.add_argument('--attr_file', default='', type=str,
                        help='attribute file')
    parser.add_argument('--training_percent', default='0.1', type=str,
                        help='training percentage, not for compute, just for save file')
    parser.add_argument('--dataset', default='', type=str,
                        help='data name, not for compute, just for save file')
    parser.add_argument('--attr_alpha', default=0.0001, type=float,
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



