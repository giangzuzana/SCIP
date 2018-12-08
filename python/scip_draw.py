#!/usr/bin/env python2
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
Created on Mon Jan 23 11:18:28 2017

@author: giang nguyen
"""
import numpy as np
import pandas as pd
import math
import itertools
from pprint import pprint
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt; 
import seaborn as sns
sns.set_context("paper")
sns.set(style="whitegrid", color_codes=True)

dataset = 'test2'
_PATH_  = './data/' + dataset + '/'
statdir = './stat_' + dataset + '/'

def draw_len_freq(fni=statdir +'taxo_cartsize_2_n.tsv', fno=statdir +'taxo_cartsize.pdf'):
    pp = PdfPages(fno)
    plot1 = plt.figure()
    plt.rcParams["figure.figsize"] = [5, 3]
    
    # df = rdd.toDF(['len', 'freq']).toPandas()         #RDD to Pandas df
    df = pd.read_csv(fni, sep='\t', skipfooter=0, engine='python', names=['size', 'freq'])
    df.sort_values(by='size', ascending=True, inplace=True)
    
    attr = df['size']
    y_pos = np.arange(len(attr))
    x_pos = df['freq']

    plt.barh(y_pos, x_pos, align='center', alpha=0.65)
    plt.axis('tight')
    plt.yticks(y_pos, attr)
    plt.ylabel('cartsize')
    plt.xlabel('freq')
    plt.title('dataset=TEST1 cartsize TAXONOMY')

    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()
    pp.savefig(plot1)
    pp.close()
    
def merge_predajnost(fn=statdir +'tovar_freq.tsv'):        #test2
    fn0 = statdir + 'dwh.tsv'
    fn1 = statdir + 'tovar_freq_2_7.tsv'
    fn2 = statdir + 'tovar_freq_2_7_since2011.tsv'   
    fn3 = statdir + 'tovar_freq_2_7_since2012.tsv'
    fn4 = statdir + 'tovar_freq_2_7_since2013.tsv'
    fn5 = statdir + 'tovar_freq_2_7_since2014.tsv'
    fn6 = statdir + 'tovar_freq_2_7_since2015.tsv'
    
    df0 = pd.read_csv(fn0, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'nazov', 'popis'])
    df1 = pd.read_csv(fn1, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'freq_all'])
    df2 = pd.read_csv(fn2, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'fs_2011'])    
    df3 = pd.read_csv(fn3, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'fs_2012'])
    df4 = pd.read_csv(fn4, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'fs_2013'])
    df5 = pd.read_csv(fn5, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'fs_2014'])
    df6 = pd.read_csv(fn6, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'fs_2015'])
    
    df = df0.merge(df1, on='tovar', how='left')\
            .merge(df2, on='tovar', how='left')\
            .merge(df3, on='tovar', how='left')\
            .merge(df4, on='tovar', how='left')\
            .merge(df5, on='tovar', how='left')\
            .merge(df6, on='tovar', how='left')
                   
    df[['freq_all', 'fs_2011', 'fs_2012', 'fs_2013', 'fs_2014', 'fs_2015']] = \
         df[['freq_all', 'fs_2011', 'fs_2012', 'fs_2013', 'fs_2014', 'fs_2015']].fillna(0.0).astype(int)
    df = df[['freq_all', 'fs_2011', 'fs_2012', 'fs_2013', 'fs_2014', 'fs_2015', 'tovar', 'nazov', 'popis']]
    print(df.dtypes)
    df.to_csv(fn, sep='\t', index=False)
    
#Tukey's test applied for list_of_freq of cart_size   
def tukey_outlier_cartsize(k=3, fn = statdir +'cartsize.tsv'):   
    df = pd.read_csv(fn, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['cartsize', 'freq'])
    size = df['cartsize'].values.tolist()[1:]          # without cart_size==1
    freq = df['freq'].values.tolist()[1:]
    n = sum(freq)
    i1 = 0.25*(n-1)
    i2 = 0.75*(n-1)    
    lower = i1
    for x in freq:
        lower = lower - x
        if lower < 0:
            index = freq.index(x)
            q1 = size[index]
            print '=====> q1=', q1, 'index=', freq.index(x)  
            break 
    upper = i2
    for x in freq:
        upper = upper - x
        if upper < 0:
            index = freq.index(x)
            q2 = size[index]
            print '=====> q2=', q2, 'index=', freq.index(x)
            break
    iqr = q2 - q1
    out = [math.floor(q1 - k*iqr), math.ceil(q2 + k*iqr)]
    print '=====> k=', k, 'outlier=', out       # k= 3 outlier= [-1.0, 6.0]
    return out

def csv_to_dict(fni= statdir +'prob.tsv', fno='./prob_tmp.tsv'):
    df = pd.read_csv(fni, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'freq', 'prob'])
    '''
    df = df[['tovar', 'freq']]
    total = df['freq'].sum()
    df['prob'] = df['freq']/total 
    df.sort_values(by=['prob'], ascending=False, inplace=True)
    df.to_csv(fno, sep='\t', index=False)
    print(df.dtypes)
    '''
    d = df.set_index('nazov').T.to_dict('list')
    for k in d:
        print k, ':', d[k]
    return d        

def read_dwh(fn_dwh=statdir + 'dwh.tsv'):
    df_dwh = pd.read_csv(fn_dwh, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'nazov', 'popis'])    
    df_dwh = df_dwh[['tovar', 'nazov']]
    print(df_dwh.dtypes)
    return df_dwh
          
def sort_pair_dwh(fni=statdir +'pair_freq_0007_40989.tsv', fno='./pair_007.tsv'):
    df_dwh = read_dwh()
 
    df = pd.read_csv(fni, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['A', 'B', 'freq'])
    df = df.merge(df_dwh, how='left', left_on=['A'], right_on=['tovar'])
    df = df.drop('tovar', axis=1).rename(columns = {'nazov':'nazov_A'})
    df = df.merge(df_dwh, how='left', left_on=['B'], right_on=['tovar'])
    df = df.drop('tovar', axis=1).rename(columns = {'nazov':'nazov_B'})

    df.sort_values(by='freq', ascending=False, inplace=True)
    print(df.dtypes)  
    df.to_csv(fno, sep='\t', index=False)
    return

def get_taxo(fno=statdir +'taxo.tsv'):    
    df = pd.read_csv('./data/taxonomia-data.tsv', sep='\t', skiprows=0, skipfooter=0, engine='python') 
    df['category'] =  df['cat0.id'].astype(str).str.rjust(2,'0') +\
                      df['cat1.id'].astype(str).str.rjust(2,'0')
    df['TOVAR'] = df['TOVAR'].astype(str)   
    #dfd = df[['TOVAR', 'category']] 
    #dtc = dfd.set_index('TOVAR').to_dict()['category']      
    dfn = df[['category', 'cat1.name']].drop_duplicates()
    #dfn.to_csv(fno, sep='\t', index=False, header=False)
    return dfn

def merge_pair_taxo(fn=statdir +'taxo_pair_freq.tsv', fno=statdir +'./taxo_pair.tsv'):
    dft = pd.read_csv('./data/taxonomia-data.tsv', sep='\t', skiprows=0, skipfooter=0, engine='python') 
    dft['category'] = dft['cat0.id'].astype(str).str.rjust(2,'0') +\
                      dft['cat1.id'].astype(str).str.rjust(2,'0')
    dft['category'] = dft['category'].astype(str).str.rjust(4,'0') 
    dft['cat1.name']= dft['cat1.name'].astype(str).str.ljust(20,' ')                  
    dft = dft[['category', 'cat1.name']].drop_duplicates() 
                 
    df = pd.read_csv(fn, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['A', 'B', 'freq'])
    df['A'] = df['A'].astype(str).str.rjust(4,'0')
    df['B'] = df['B'].astype(str).str.rjust(4,'0')
    df = df.merge(dft, how='left', left_on=['A'], right_on=['category'])
    df = df.drop('category', axis=1).rename(columns = {'cat1.name':'cat1_A'})
    df = df.merge(dft, how='left', left_on=['B'], right_on=['category'])
    df = df.drop('category', axis=1).rename(columns = {'cat1.name':'cat1_B'})
    '''       
    with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
        print df    
    '''
    df.sort_values(by='freq', ascending=False, inplace=True)
    print(df.dtypes)    
    df.to_csv(fno, sep='\t', index=False)    
    return
    
def merge_stat(fni, fno):
    df_dwh = read_dwh()
    df_fni = pd.read_csv(fni, sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'freq'])
    print(df_fni.dtypes)
    
    df = df_fni.merge(df_dwh, how='left', on=['tovar'])   
    print(df.dtypes)    
    df.to_csv(fno, sep='\t', index=False) 

def subset(arr, len_max=6):
    if len(arr) < len_max:
        loss = []
        for i in range(1, len(arr)+1):
            for subset in itertools.combinations(arr, i):
                loss.append(subset)
    else:
        loss = [ arr ]        
    return loss

def pokladna_only(statdir, fno):
    df1 = pd.read_csv(statdir +'pokladna1.tsv', sep='\t', skiprows=0, skipfooter=0, engine='python')
    df2 = pd.read_csv(statdir +'pokladna2.tsv', sep='\t', skiprows=0, skipfooter=0, engine='python')
    df3 = pd.read_csv(statdir +'pokladna3.tsv', sep='\t', skiprows=0, skipfooter=0, engine='python')
    l1 = df1['tovar'].values
    l2 = df2['tovar'].values    
    l3 = df3['tovar'].values
    
    s = set(l3) - set(l2) - set(l1)
    dfs = pd.DataFrame(list(s), columns=['tovar'])
    df = dfs.merge(df3, how='left', on=['tovar'])
    
    df.sort_values(by='freq', ascending=False, inplace=True)
    df.to_csv(fno, sep='\t', index=False)
    print len(s)

def test2_taxo(fno=statdir+'test2_taxo_dwh.tsv'):
    df1 = pd.read_csv(statdir +'test2_dwh.tsv',  sep='\t', skiprows=0, skipfooter=0, engine='python', names=['tovar', 'nazov', 'popis'])
    df2 = pd.read_csv(statdir +'test2_taxo.txt', sep=',',  skiprows=0, skipfooter=0, engine='python', names=['cat0.id', 'cat1.id', 'tovar'])
    df = df1.merge(df2, on='tovar', how='left')
    df['cat0.name'] = 'cn0_' + df['cat0.id'].astype(str)
    df['cat1.name'] = 'cn1_' + df['cat1.id'].astype(str)
    df['SKLAD'] = '0'
    df = df[['cat0.id', 'cat1.id', 'cat0.name', 'cat1.name', 'SKLAD', 'tovar', 'nazov', 'popis']]
    df.rename(columns={'tovar': 'TOVAR', 'nazov': 'NAZOV', 'popis': 'POPIS'}, inplace=True)
    df.sort_values(['cat0.id', 'cat1.id', 'TOVAR'], ascending=True, inplace=True)
    df.to_csv(fno, sep='\t', index=False, header=True)
    
def main(argv):
    #draw_len_freq(statdir +'taxo_cartsize_2_n.tsv', statdir +'taxo_cartsize_2_n.pdf')
    #tukey_outlier_cartsize(3, statdir +'cartsize.tsv')
    #merge_pair_taxo()    
    #merge_predajnost()
    #csv_to_dict()
    #sort_pair_dwh()
    #get_taxo()
    #merge_stat(statdir +'freq_item.tsv', statdir +'freq_item_dwh.tsv')
    #pokladna_only(statdir, statdir +'pokladna3-only-tovar.tsv')
    print subset([1, 2, 3, 4], len_max=6)
    test2_taxo()
    
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
    
