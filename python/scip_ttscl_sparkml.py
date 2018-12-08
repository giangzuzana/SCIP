# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Created on Fri Dec  9 13:55:58 2016
@author: giang nguyen
"""
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from pyspark import SparkContext
from pyspark.sql import SparkSession
#from pyspark.sql import Row
from pyspark.ml import Pipeline
#from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime, timedelta
from operator import itemgetter
from pprint import pprint
import csv
import itertools
import math
import pandas as pd
import random
import time

spark = SparkSession \
    .builder \
    .appName('GN_datapac_slovnaft_data_analysis') \
    .config('spark.driver.memory', '24G') \
    .config('spark.executor.memory', '24G') \
    .getOrCreate()
sc = spark.sparkContext

# global parameters
numrows = 200
dataset = 'test2'
datadir = './data/'
_PATH_  =   datadir + dataset + '/'
statdir = './stat_' + dataset + '/'

taxo   = True               # use taxonomy OR tovar
subset = True               # NB with subset generator for every cart
unique = False              # rdd_cart.distinct()
pokladna = 'all'    # shop OR espresso OR shop1 OR shop2 OR all
cartsize_min = 2            # test1: 2<= cartsize <=6, 37
cartsize_max = 6            # test2: 2<= cartsize <=6, 42
topn = 5                # topn recommendations
smooth_k = 0.01         # k-smooth for Multinomial Naive Bayes (NB)
support_item = 0.01         # has to be 0.1
support_pair = 0            # has to be 0.1
confidence_threshold = 0    # has to be 0.5
lift_threshold = 0          # has to be 1.0
timerange = []                                # whole time
#timerange = ['2015-01-01', '2016-01-01']
#timerange = ['2014-08-28', '2016-11-08']     # test1
#timerange = ['2010-09-29', '2016-11-08']     # test2 
     
# tables = ['F_UCTY_H', 'F_UCTY_L', 'F_SKLAD']
# F_UCTY_L niektore polozky maju zapornu cenu (storno) 

def run_time(start, format_str='%Y%m%d-%H%M%S'):
    diff = time.mktime(datetime.strptime(time.strftime(format_str), format_str).timetuple()) - \
           time.mktime(datetime.strptime(start, format_str).timetuple())
    return str(timedelta(seconds=diff)) + 's'
    
# from Steva :) Spark 2.0
def read_table(table, ignore_first_ucet=0):
    global spark, sc, _PATH_, numrows
    df = spark.read.parquet(_PATH_ + table + '.parquet')
    df.createOrReplaceTempView(table)
    if ignore_first_ucet==0:
        q = 'SELECT * FROM ' + table + ' LIMIT ' + str(numrows)
        # q = 'SELECT * FROM ' + table + ' WHERE pokladna = 3 LIMIT ' + str(numrows)
    else: 
        q = 'SELECT * FROM ' + table + ' WHERE ucet > ' + str(ignore_first_ucet) + ' LIMIT ' + str(numrows)
    df = spark.sql(q)
    print '\n=========> read_table(): count=', table, df.count()
    df.show(numrows, False)

# raw: dlzka uctov (c) vs freq (f) bez redukovania opakovanych tovarov a storna
def count_freq_ucet(table='F_UCTY_L'):
    global spark, sc, _PATH_, numrows
    df = spark.read.parquet(_PATH_ + table + '.parquet')
    df.createOrReplaceTempView(table)
    if table == 'F_UCTY_L':
        q = 'SELECT *, COUNT(*) AS f FROM \
                (SELECT COUNT(*) AS c FROM ' + table + ' GROUP BY pokladna, ucet ) \
            GROUP BY c ORDER BY f DESC'
    elif table == 'F_UCTY_H':
        q = 'select *, count(*) as f from \
                (select count(*) as c from ' + table + ' group by pokladna, suma_pc) \
            group by c order by f desc'
    myrdd = spark.sql(q).rdd                     
    #draw_len_freg(myrdd, './test1_ucty_len_freq.pdf')
    for x in myrdd.collect():
        print x.c, '\t', x.f
        
def read_datumcas(timerange=['2014-01-01', '2017-01-01'], table='F_UCTY_H'):
    global spark, sc, _PATH_, numrows
    dt_min = datetime.strptime(timerange[0], '%Y-%m-%d')
    dt_max = datetime.strptime(timerange[1], '%Y-%m-%d')
    
    df = spark.read.parquet(_PATH_ + table + '.parquet')
    df.createOrReplaceTempView(table)
    q = 'SELECT pokladna, ucet, datumcas FROM ' + table
    rdd_time = (spark.sql(q).rdd 
        .map(lambda row: ( str(row.pokladna) +'_'+ str(row.ucet), row.datumcas ) ) 
        .filter(lambda row: ( dt_min < row[1] < dt_max ) )	             
        .cache() 
        )    
    print '\n==========> read_datumcas(): rdd_time.count()=', rdd_time.count()
    print 'min=', rdd_time.values().min(), '\nmax=', rdd_time.values().max()
    return rdd_time
     
# dict {tovar: nazov}
def get_dwh(fno=statdir +'dwh.tsv'):
    global spark, sc, _PATH_, numrows
    table='F_SKLAD'
    
    df = spark.read.parquet(_PATH_ + table + '.parquet')
    df.createOrReplaceTempView(table)
    q = 'SELECT tovar, nazov, popis FROM ' + table + ' DESC'
    df = spark.sql(q).distinct()  
    print '\n==========> get_dwh(): F_SKLAD pocet tovarov=', df.count()
    
    arr = df.rdd.collect()   
    dwh = {}
    for x in arr:
        dwh[x.tovar] = x.nazov      # {tovar: nazov}
        
    if len(fno) > 0:  
        with open(fno, 'w') as f:
            for x in arr:
                f.write(x.tovar +'\t'+ x.nazov +'\t'+ x.popis +'\n')
    return dwh

# dict {category: cat1.name}
def get_dwh_taxo(fni=datadir + 'taxonomia-data.tsv'):
    df = pd.read_csv(fni, sep='\t', skiprows=0, skipfooter=0, engine='python') 
    df['category'] =  df['cat0.id'].astype(str).str.rjust(2,'0') +\
                      df['cat1.id'].astype(str).str.rjust(2,'0')
    df['cat.name'] =  df['cat0.name'].astype(str) +'-'+\
                      df['cat1.name'].astype(str)
    df = df[['category', 'cat.name']]
    d = df.set_index('category').to_dict()['cat.name']
    print '\n==========> get_taxo_dwh(): pocet kategorii=', len(d)   # 56
    return d

# dict {tovar: category}
def get_taxo(fni=datadir + 'taxonomia-data.tsv'):
    df = pd.read_csv(fni, sep='\t', skiprows=0, skipfooter=0, engine='python') 
    df['category'] =  df['cat0.id'].astype(str).str.rjust(2,'0') +\
                      df['cat1.id'].astype(str).str.rjust(2,'0')
    df['TOVAR'] = df['TOVAR'].astype(str)
    df = df[['TOVAR', 'category']] 
    d  = df.set_index('TOVAR').to_dict()['category']
    print '\n==========> get_taxo(): pocet tovarov=', len(d)        # 2828 
    return d
    
def print_carts(rdd_cart, fno=statdir + 'carts.tsv'):
    #d = get_dwh()              # {tovar: nazov}
    #d = get_taxo()             # {tovar: category}
    U = rdd_cart.collect()
    with open(fno, 'w') as f:
        for cart in U:
            for x in cart:
                f.write(x +'\t')                         # tovar
                #f.write(d[x].ljust(16)[:16] +'  ')      # nazov/category
            if cart:
                f.write('\n')

def get_cartsizes(rdd_cart, cartsize_min=1, fn=statdir + 'cartsize.tsv'):
    U = rdd_cart.collect()
    d_size = {}                          # {cartsize: freq} 
    for cart in U:
        cartsize = len(cart)
        if cartsize >= cartsize_min:
            if cartsize in d_size:
                d_size[cartsize] += 1
            else:
                d_size[cartsize] = 1 
             
    with open(fn, 'w') as f:
        for cartsize, freq in sorted(d_size.items(), key=itemgetter(1), reverse=True):
            f.write(str(cartsize) +'\t'+ str(freq) +'\n')
    return d_size

# cart = ( pokladna_ucet, [ [tovar, mnozstvo],    [tovar, mnozstvo], ... ] )  
# cart = ( pokladna_ucet, [ [category, mnozstvo], [category, mnozstvo], ... ] )          
def map_cart(row_cart, dtaxo, palivo='0101'):
    d = {}
    for x in row_cart:
        tovar    = x[0]
        mnozstvo = x[1]
        if tovar in d:
            d[tovar] += mnozstvo
        else:
            d[tovar] = mnozstvo
    cart = []
    for tovar, mnozstvo in d.iteritems():
        if mnozstvo > 0:
            if dtaxo == None:
                cart.append(tovar)
            else:
                if tovar in dtaxo:
                    cart.append(dtaxo[tovar])
                else:
                    cart.append('0000')         # tovar not in taxonomy
    cart = tuple(set(sorted(cart)))
    #if len(palivo) > 0:                     
    #   if palivo in cart: cart.remove(palivo)
    return cart
 
def dp_carts(taxo, timerange, cartsize_min=1, cartsize_max=37, pokladna='shop', unique=False, 
             fno=statdir + 'carts.tsv'):
    global spark, sc, _PATH_, numrows 
    dtaxo = None
    if taxo:
        dtaxo = get_taxo()                      # {tovar, category} from taxonomy          
          
    table = 'F_UCTY_L'
    df = spark.read.parquet(_PATH_ + table + '.parquet')
    df.createOrReplaceTempView(table)
    q = 'SELECT pokladna, ucet, tovar, mnozstvo FROM ' + table
     
    rdd_cart = spark.sql(q).rdd
    if   pokladna == 'shop':
        rdd_cart = rdd_cart.filter(lambda row: row.pokladna != 3)
    elif pokladna == 'espresso':
        rdd_cart = rdd_cart.filter(lambda row: row.pokladna == 3)
    elif pokladna == 'shop2':
        rdd_cart = rdd_cart.filter(lambda row: row.pokladna == 2)
    elif pokladna == 'shop1':
        rdd_cart = rdd_cart.filter(lambda row: row.pokladna == 1)
        
    rdd_cart = ( rdd_cart    
        .map(lambda row: ( str(row.pokladna) +'_'+ str(row.ucet), [ [row.tovar, row.mnozstvo] ]) ) 
        .reduceByKey(lambda x, y: x + y) 
        .map(lambda (key, cart): (key, map_cart(cart, dtaxo) ) ) 
        .filter(lambda (key, cart): cartsize_min <= len(cart) <= cartsize_max )   
        .cache()
        )
    if timerange:
        rdd_datumcas = read_datumcas(timerange, 'F_UCTY_H')
        rdd_cart = ( rdd_cart
            .leftOuterJoin(rdd_datumcas)
            .filter(lambda (key, (cart, datumcas)): datumcas != None)
            .map(lambda (key, (cart, datumcas)): cart) ) 
    else:
        rdd_cart = rdd_cart.map(lambda (key, cart): cart)
        
    if unique:
        rdd_cart = rdd_cart.distinct()
    print '\n==========> dp_carts(): cart_num=', rdd_cart.count(), 'unique_carts=', unique
    
    if len(fno) > 0:
        print_carts(rdd_cart, fno)
    return rdd_cart

# cart = [tovar1, tovar2, ...]
def get_freq(rdd_cart, fno=statdir +'freq_item.tsv'):
    rdd_freq = ( rdd_cart
        .flatMap(lambda x: x)
        .map(lambda x: (x, 1)) 
        .reduceByKey(lambda x, y: x + y)
        .sortBy(lambda x: x[1], False)
           )
    if len(fno) > 0:
        with open(fno, 'w') as f:
            for tovar, freq in rdd_freq.collect():
                f.write(tovar +'\t'+ str(freq) +'\n')
    print '\n==========> get_freq(): count= ', rdd_freq.count() 
    return rdd_freq

# long running: another implementation on Spark/BigData !!! 
def get_freq_pair(rdd_cart, rdd_freq, fno=statdir +'freq_pair.tsv'):
    global support_item
    threshold = float( rdd_cart.count() * support_item )
    print '\n==========> get_freq_pair(): support_item >', threshold
        
    rdd = ( rdd_freq
        .filter(lambda (X, freq): freq > threshold)         # filter support_item
        .sortBy(lambda (X, freq): X)
        .map(lambda (X, freq): X)
            )
    rdd = ( rdd
        .cartesian(rdd)
        .filter(lambda (A, B): A < B )                      # A < B only
            )
    with open(fno, 'w') as f:
        i = 0
        for pair in rdd.collect():
            i += 1
            print i,  
            freq = rdd_cart.filter(lambda cart: len(set(pair).intersection(cart)) == 2).count()
            if freq > 0:
                f.write(pair[0] +'\t'+ pair[1] +'\t'+ str(freq) +'\n')
    print '\n==========> get_freq_pair(): cartesian count = ', rdd.count() 

def median(arr):
    arr = sorted(arr)
    if len(arr) < 1:
        return None
    if len(arr) %2 == 1:
        return arr[((len(arr)+1)/2)-1]
    else:
        return float(sum(arr[(len(arr)/2)-1:(len(arr)/2)+1]))/2.0
        
def format(value):
    return "%10.5f" % value

# interaction of tovar with other tovars
def interaction(taxo, cart_num, fni=statdir +'freq_pair.tsv', fno=statdir +'interaction.tsv'):
    global support_pair
    threshold = float( cart_num * support_pair)
                      
    rdd = sc.textFile(fni).map(lambda line: line.split('\t'))                    
    header = rdd.first()
    rdd = ( rdd
        .filter(lambda row: row != header)
        .map(lambda x: ( x[0], x[1], int(x[2]) ) )
        .filter(lambda (A, B, freq): freq > threshold )
            )
    iA  = ( rdd 
        .map(lambda (A, B, freq): (A, [(B, freq)] ) )
        .reduceByKey(lambda x, y: x + y)
           )
    iB  = ( rdd 
        .map(lambda (A, B, freq): (B, [(A, freq)] ) )
        .reduceByKey(lambda x, y: x + y)
           )
    iAB = (iA.union(iB)
        .reduceByKey(lambda x, y: x + y)
        .map(lambda (tovar, iterlist): (tovar, [ freq for [t, freq] in iterlist ]))    
            )    
    with open(fno, 'w') as f:
        if not taxo:
            d = get_dwh('')                  # {tovar: nazov}
        else:
            d = get_dwh_taxo()               # {category: cat1.name}
        f.write('nazov' +'\t'+ 'item' +'\t'+ 'n' +'\t'+ 'max' +'\t'+ 'avg' +'\t'+ 'median' +'\t'+ 'std.dev' +'\t'+ 'list' +'\n')
        for (item, freqlist) in iAB.collect():
            n     = len(freqlist)
            fmax  = max(freqlist)
            fmean = sum(freqlist)/float(n)
            fmed  = median(freqlist)
            fsd    = math.sqrt(sum((x - fmean)**2 for x in freqlist) / n)
            f.write(d[item] +'\t'+ item +'\t'+ str(n) +'\t'+ str(fmax) +'\t'+ format(fmean) +'\t'+ str(fmed) +'\t'+ format(fsd) +'\t')
            f.write(' '.join(str(freq) for freq in sorted(freqlist, reverse=True)[:100] )  +'\n')
    return

# https://en.wikipedia.org/wiki/Additive_smoothing
def smooth_prob(smooth_k, cart_num, n_item, item, freq):
    if smooth_k > 0:
        prob = float(freq + smooth_k) / (cart_num + smooth_k * n_item)
    else:
        prob = float(freq) / cart_num
    return (item, prob)

def smooth_prob_pair(smooth_k, cart_num, n_pair, A, B, freq):
    if smooth_k > 0:
        prob = float(freq + smooth_k) / (cart_num + smooth_k * n_pair)
    else:
        prob = float(freq) / cart_num
    return (A, B, prob)
    
# http://paginas.fe.up.pt/~ec/files_1112/week_04_Association.pdf
def confidence_ab(smooth_k, cart_num, rdd_freq=None, fn_freq_item=statdir +'freq_item.tsv', 
                                                     fn_freq_pair=statdir +'freq_pair.tsv',  
                                                     fno=statdir +'confidence_ab.tsv'):
    global support_item
    global support_pair
    global confidence_threshold
    global lift_threshold
    threshold_item = cart_num * support_item
    threshold_pair = cart_num * support_pair
    print '\n==========> confidence_ab(): smooth_k=', smooth_k, 'support_pair >', threshold_pair, 'support_item >', threshold_item
    
    if rdd_freq == None:
        rdd_freq = ( sc.textFile(fn_freq_item)          # 2545
            .map(lambda line: line.split('\t'))  
            .map(lambda x: ( x[0], float(x[1]) ) )
            )
    n_item = rdd_freq.count()
    d = ( rdd_freq
        .map(lambda (item, freq): smooth_prob(smooth_k, cart_num, n_item, item, freq))
        .collectAsMap()                                 # { item: prob }
            )
    rdd_prob = ( sc.textFile(fn_freq_pair)
        .map(lambda line: line.split('\t'))                       
        .map(lambda x: ( x[0], x[1], float(x[2]) ) )    # ( A, B, freq )
            )
    n_pair = rdd_prob.count()
    rdd_prob = ( rdd_prob 
        .filter(lambda (A, B, freq): freq > threshold_pair )                    # filter support 
        .map(lambda (A, B, freq): smooth_prob_pair(smooth_k, cart_num, n_pair,  A, B, freq) )
        .map(lambda (A, B, prob): [ (A, B, prob/d[B]), (B, A, prob/d[A]) ] )    # AB & BA confidence
        .flatMap(lambda x: x)
        .map(lambda (A, B, conf): (A, B, conf, conf/d[A]) )                     # AB_lift = BA_lift
        .filter(lambda (A, B, conf, lift): lift > lift_threshold)               # filter lift
        .filter(lambda (A, B, conf, lift): conf > confidence_threshold)         # filter confidence
        .sortBy(lambda (A, B, conf, lift): (conf, lift),  False)
            )
    if len(fno) > 0:
        with open(fno, 'w') as f:
            f.write('A' +'\t'+ 'B' +'\t'+ 'confidence' +'\t'+ 'lift' +'\n')
            for (A, B, conf, lift) in rdd_prob.collect():
                f.write(A +'\t'+ B +'\t'+ format(conf) +'\t'+ format(lift) +'\n')
    return rdd_prob

# lex = list of empirical items X
def empirical_x(rdd_prob):
    lex = ( rdd_prob
        .map(lambda (A, B, conf, lift): (A, B))
        .flatMap(lambda x: x)
        .distinct()
        .collect() )
    return lex

# dax = dict of empirical prob_ab
def empirical_ab(rdd_prob):
    dprob  = ( rdd_prob
        .map(lambda (A, B, conf, lift): ( (str(A) +'_'+ str(B)), conf ) )
        .collectAsMap() )
    return dprob

# classic Naive Bayes (NB) + topn recommendation     cart = [tovar1, tovar2, ...] 
def nb_classic(cart,lex, dax, dxx):
    lex_without_cart = [item for item in lex if item not in cart]
    dnb = {}
    for X in lex_without_cart:
        for A in cart:
            key = str(A) +'_'+ str(X)
            if key in dax:
                if X in dnb:
                    dnb[X] *= dax[key]
                else:
                    dnb[X] = dxx[X] * dax[key]         # with P(X)
    recomm = [x[0] for x in sorted(dnb.items(), key=lambda x: x[1], reverse=True) ]     # list of X
    return recomm

# S = list of subsets - generate all subset combinations of the cart
def generate_subsets(cart, len_max=6):
    if len(cart) <= len_max:
        S = []
        for i in range(1, len(cart)+1):
            for subset in itertools.combinations(cart, i):
                S.append(subset)
    else:
        S = [ cart ]
    return S

# NB + subset
def nb_subset(cart,lex, dax, dxx):
    lex_without_cart = [item for item in lex if item not in cart]   # list of empirical items without cart
    dnb = {}
    for X in lex_without_cart:
        dnb[X] = 0
        nb_X   = 0
        for c in generate_subsets(cart):
            nb_X = 0
            for A in c:
                key = str(A) +'_'+ str(X)
                if key in dax:
                    if nb_X == 0:
                        nb_X = dxx[X] * dax[key]    # with P(X)
                    else:
                        nb_X *= dax[key]
        if nb_X > dnb[X]:
            dnb[X] = nb_X                                          
    recomm = [x[0] for x in sorted(dnb.items(), key=lambda x: x[1], reverse=True) ]     # list of X
    return recomm

# NB with topn recommendations
def nb(cart,lex, dax, dxx, topn, subset=True):
    if subset:
        recomm = nb_subset(cart,lex, dax, dxx)
    else:
        recomm = nb_classic(cart,lex, dax, dxx)
    return recomm[:topn]

    
def safe_div(x, y):
    return float(x)/float(y) if y else 0

def eval_nb(taxo, rdd_prob, rdd_cart=None, rdd_freq=None, unique=False,
                            fn_carts=statdir +'carts.tsv', fn_freq_item=statdir +'freq_item.tsv', 
                            fno=statdir +'eval_nb.tsv'):
    global topn, subset

    if rdd_cart == None:
        rdd_cart = ( sc.textFile(fn_carts)
            .map(lambda line: line.split('\t') ) )  
    cart_num = rdd_cart.count()
    
    if rdd_freq == None:
        rdd_freq = ( sc.textFile(fn_freq_item)
            .map(lambda line: line.split('\t'))  
            .map(lambda x: ( x[0], float(x[1])/cart_num) ) )
    dxx = rdd_freq.collectAsMap()       # dict of prob_X 
    lex = empirical_x( rdd_prob)        # list of empirical items X = frequent item set
    dax = empirical_ab(rdd_prob)        # dict of empirical prob_AX = frequent binary rules 
    print '\n==========> eval_nb(): lex count=', len(lex)
     
    rdd = sc.emptyRDD()
    for X in lex: 
        rddX = ( rdd_cart  
            .filter(lambda cart: X in cart) 
            .map(lambda cart: [item for item in cart if item != X] )
            .filter(lambda cart: len(cart) > 0 )
            .map(lambda cart: tuple(sorted(cart)) )
            .map(lambda cart: (X, cart) )
                )
        rdd = rdd.union(rddX)
    if unique:
        rdd = rdd.distinct()
    rdd = ( rdd                            # NB with topn recommendations
        .map(lambda (label, cart): (label, nb(cart, lex, dax, dxx, topn, subset) ) )    
        .cache()
            ) 
    n = rdd.count()
    print '\n==========> eval_nb(): testing dataset count=', n
    i = 0
    tp_all = 0
    result = {}            
    for X in lex: 
        i += 1
        print i,
        rddX = rdd.filter(lambda (label, recomm): label == X)
        nX   = rddX.count()
        tp_arr = ( rddX     
            .filter(lambda (label, recomm): X in recomm) 
            .map(lambda (label, recomm): recomm.index(X) )
            .map(lambda index: (index, 1) )
            .reduceByKey(lambda x, y: x + y)
            .sortBy(lambda (index, freq): index)
            .map(lambda (index, freq): freq)
            .collect()
                )
        tp = sum(tp_arr)
        tp_all += tp   
        result[X] = [n, nX, tp, float(tp)/nX] + tp_arr
        '''
        tp = rdd.filter(lambda (label, recomm): label == X and X     in recomm).count() 
        tn = rdd.filter(lambda (label, recomm): label == X and X not in recomm).count() 
        fp = rdd.filter(lambda (label, recomm): label != X and X     in recomm).count() 
        fn = rdd.filter(lambda (label, recomm): label != X and X not in recomm).count() 
        accuracy  = safe_div(tp + tn, n)
        precision = safe_div(tp, tp + fp)
        recall    = safe_div(tp, tp + fn)
        f1 = safe_div( 2*precision*recall, precision + recall )        
        result[X] = [accuracy, precision, recall, f1] + [n, nX, tp, float(tp)/nX] + tp_arr
        '''              
    acc = safe_div(tp_all, n)
    mae = safe_div(n - tp_all, n)
    #rmse = math.sqrt(mae)         
    print '\n==========> eval_nb(): overall_acc=', acc, '\tMAE=', mae         
    if len(fno) > 0:
        if not taxo:
            d = get_dwh('')                  # {tovar: nazov}
        else:
            d = get_dwh_taxo()               # {category: cat0.name+cat1.name}
        with open(fno, 'w') as f:
            wr = csv.writer(f, delimiter='\t')
            for (X, r) in sorted(result.items(), key=lambda x: x[1][3], reverse=True):
                row = [ d[X], X ] + r
                wr.writerow(row)
    return 
   
# libsvm dense format <label> <index1>:<value1> <index2>:<value2> ...
# one-based indices, ascending order of indexes, without subsets, unique, shuffle
def libsvm_data( rdd_cart=None, rdd_freq=None, unique=False,
                      fn_carts=statdir +'carts.tsv', fn_freq_item=statdir +'freq_item.tsv',
                      fno=statdir +'libsvm_data.txt'):
    global support_item    
    if rdd_cart == None:
        rdd_cart = ( sc.textFile(fn_carts)
            .map(lambda line: line.split('\t') ) 
            )       
    if rdd_freq == None:
        rdd_freq = ( sc.textFile(fn_freq_item)
            .map(lambda line: line.split('\t'))  
            .map(lambda line: (line[0], int(line[1]) ) )
            )
    threshold = float( rdd_cart.count() * support_item )
    rdd_freq = ( rdd_freq
        .sortBy(lambda (item, freq): freq)
        .filter(lambda (item, freq): freq > threshold)      # filter support_item
        .map(   lambda (item, freq): item)
            )
    lex = rdd_freq.collect()
    print '\n==========> libsvm_data(): freq items len(lex)=', len(lex), '\t support_item >', threshold
    
    numX = []
    if len(fno) > 0:
        with open(fno, 'w') as f:
            rdd = sc.emptyRDD()
            for X in lex: 
                rddX = ( rdd_cart 
                    .filter(lambda cart: X in cart) 
                    .map(lambda cart: [lex.index(item)+1 for item in cart if item != X and item in lex] )
                    .filter(lambda cart: len(cart) > 0 )
                    .map(lambda cart: tuple(sorted(cart)) )
                    .map(lambda cart: (X, cart) )
                        )
                if unique:
                    rddX = rddX.distinct() 
                rdd = rdd.union(rddX)  
                numX.append(rddX.count())
                
            for (X, cart) in rdd.collect():
                line  = str(lex.index(X) +1) +' '
                line += ' '.join((str(x) +':1') for x in cart)
                f.write(line +'\n')
    lines = open(fno, 'r').readlines() 
    random.shuffle(lines)
    open(fno, 'w').writelines(lines)
    print '\n==========> libsvm_data() lines=', rdd.count(), '\n', numX
    return

# este nefunguju ... TODO
def spark_ml(fni=statdir +'libsvm_data.txt', fno=statdir +'ml_prediction.txt'):
    data  = spark.read.format('libsvm').load(fni)
    splits= data.randomSplit([0.8, 0.2], 12345)
    train = splits[0]
    test  = splits[1] 
    #nb = NaiveBayes(smoothing=0.01, modelType="multinomial")
    #model = nb.fit(train)
    labelIndexer   = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    model = pipeline.fit(train)
    pred  = model.transform(test)      # predictions   
    #pred.printSchema() 
    #pred.select('label', 'prediction').show(numrows)      
    pred.createOrReplaceTempView('pred')
    q = 'SELECT * FROM \
            ( SELECT label, prediction, COUNT (*) as c FROM pred GROUP BY label, prediction )\
         WHERE label=prediction ORDER BY label'
    df = spark.sql(q)
    rdd = df.rdd   
    total = 0                  
    for row in rdd.collect():
        total += row.c
        print row.label, '\t', row.prediction, '\t', row.c       
    print '\n==========> spark_ml(): n=', data.count(), 'tpn=', pred.count(), 'tp=', total
        
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    acc = evaluator.evaluate(pred)    
    print '\n==========> spark_ml(): Test set acc = ', str(acc)
    return


def main(argv):
    start = time.strftime("%Y%m%d-%H%M%S")
    #dwh = get_dwh(statdir + 'dwh.tsv')           # {tovar: nazov}
    #rs = 1.0/float(dwh.count())                  # random selection 1/3069=0.000333 

    rdd_cart = dp_carts(taxo, timerange, cartsize_min, cartsize_max, pokladna, unique, statdir +'carts.tsv')     
    cart_num = rdd_cart.count()
    rdd_freq = get_freq(rdd_cart,     statdir +'freq_item.tsv')        
    get_freq_pair(rdd_cart, rdd_freq, statdir +'freq_pair.tsv')         
    rdd_prob = confidence_ab(smooth_k, cart_num, rdd_freq, statdir +'freq_item.tsv', statdir +'freq_pair.tsv', statdir +'confidence_ab.tsv')
    eval_nb( taxo, rdd_prob, rdd_cart, rdd_freq, unique,   statdir +'carts.tsv',     statdir +'freq_item.tsv')
    #libsvm_data(rdd_cart, rdd_freq, unique, statdir +'carts.tsv', statdir +'freq_item.tsv') 
    #libsvm_data(None, None, unique,         statdir +'carts.tsv', statdir +'freq_item.tsv')
    #spark_ml(statdir +'libsvm_data.txt') 
    '''
    rdd_cart = dp_carts(taxo, timerange, cartsize_min, cartsize_max, pokladna, unique, statdir +'carts.tsv')     
    cart_num = rdd_cart.count()                                             # 299494
    get_cartsizes(rdd_cart, cartsize_min, statdir +'cartsize.tsv')
    rdd_freq = get_freq(rdd_cart,         statdir +'freq_item.tsv')         # 2545        
    get_freq_pair(rdd_cart, rdd_freq,     statdir +'freq_pair.tsv')         # tovar->long running->cluster?
    interaction(taxo, cart_num,           statdir +'freq_pair.tsv', statdir +'interaction.tsv')                                           
    rdd_prob = confidence_ab(smooth_k, cart_num, rdd_freq, statdir +'freq_item.tsv', statdir +'freq_pair.tsv', statdir +'confidence_ab.tsv')
    eval_nb(taxo,  rdd_prob, rdd_cart, rdd_freq, unique,   statdir +'carts.tsv',     statdir +'freq_item.tsv') 
    
    read_table('F_UCTY_L', 0)
    read_table('F_UCTY_H', 0)
    read_table('F_SKLAD')
    read_table('F_POKLADNE', 0)
    read_table('C_AKCIOVE_BALICKY_SKUPINY')
    count_freg_ucet()
    count_freg_ucet('F_UCTY_H')
    show_stat_skupina('F_SKLAD')
    dp_storno()   
    for x in rdd.take(numrows):
        print x
    '''          
    print '\n==========> Runtime=', run_time(start)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Datapac', epilog='---')
    parser.add_argument("--output",
                        default='./datapac_stats',
                        dest="outFN", help="output_file", metavar="FILENAME")
    parser.add_argument("--log",
                        default='./datapac_log',
                        dest="logFN", help="log_file", metavar="FILENAME")
    args = parser.parse_args()
    main(args)
    
