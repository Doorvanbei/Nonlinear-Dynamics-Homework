import numpy as np
import matplotlib.pyplot as plt
ecdunits, dcdunits, hlen = 48, 24, 48
tb = 0 # test bias
lr = 0.0005 # learning rate
'''
real hour-price prediction based on simple RNN
作者：陈长 
20210109
'''
def getxy(data, start, end, ecdunits, dcdunits):
    x, y = np.array([0.0]), np.array([0.0])
    for i in range(start, end):
        x = np.append(x, data[i])
        x = np.append(x, data[i + 1])
        y = np.append(y, data[i + 2])
    x, y = np.delete(x, 0), np.delete(y, 0)
    return np.reshape(x, (-1, ecdunits)), np.reshape(y, (-1, dcdunits))

def RNN(whh, wxh, why, h, x, y, ecdunits, dcdunits):
    for i in range(ecdunits):  # encode
        h[i + 1] = whh[i] @ h[i] + wxh[i] @ x[i]
        h[i + 1] = np.tanh(h[i + 1])
    for i in range(ecdunits, ecdunits + dcdunits):  # decode
        h[i + 1] = whh[i] @ h[i]
        h[i + 1] = np.tanh(h[i + 1])
        y[i - ecdunits] = why[i - ecdunits] @ h[i + 1]
    return h, y
#-------------------------- load data --------------------------
data = np.loadtxt(r'price_730ds.csv')  # shape = (17520,)
data = np.reshape(data, (-1, dcdunits))  # shape = (730, 24)
x_tr, y_tr_l = getxy(data, 0, 700, ecdunits, dcdunits)
x_va, y_va_l = getxy(data, 700, 724, ecdunits, dcdunits)
x_te, y_te_l = getxy(data, 724 - tb, 728 - tb, ecdunits, dcdunits)
men, var = np.mean(x_tr), np.var(x_tr)
x_tr, y_tr_l, x_va, y_va_l, x_te = (x_tr-men)/var,(y_tr_l-men)/var,(x_va-men)/var,(y_va_l-men)/var,(x_te-men)/var
wxh = np.tanh( np.random.normal(size=(ecdunits, hlen, 1)) )
why = np.tanh( np.random.normal(size=(dcdunits, 1, hlen)) )
whh = np.tanh( np.random.normal(size=(ecdunits + dcdunits, hlen, hlen)) )
# wxh = np.load('wxh_h48.npy')
# whh = np.load('whh_h48.npy')
# why = np.load('why_h48.npy')
h = np.zeros((ecdunits + dcdunits + 1, hlen, 1)) # state vector h initialization
y = np.zeros(dcdunits) # output vector y initialization
#-------------------------- training process --------------------------
losseps = 1.0e-9 # if loss < losseps: break
lossold = 1.0e6 # old loss initialization
valossold = 0.0010249270541722736
for epoch in range(np.int64(3e9)): # train the whole tr set once
    dl_dwhy, dl_dwhh, dl_dwxh = np.zeros_like(why), np.zeros_like(whh), np.zeros_like(wxh)
    loss_epo = np.float64(0.0)
    for smp in range(700):
        x = np.reshape(x_tr[smp], (ecdunits, 1, 1))
        h, y = RNN(whh, wxh, why, h, x, y, ecdunits, dcdunits)
        delta_y = y - y_tr_l[smp]
        loss_epo += np.mean(delta_y ** 2)
        dloss_dhlast = 0
        for i in range(-1, -(dcdunits + 1), -1):
            dloss_dythis = 2. / dcdunits * (delta_y[i])
            dloss_dwhythis = dloss_dythis * h[i].T
            dloss_dhthis = dloss_dythis * why[i].T + dloss_dhlast
            dloss_dbfhthis = dloss_dhthis * (1 - h[i] ** 2)
            dloss_dwhhthis = dloss_dbfhthis @ h[i - 1].T
            dloss_dhlast = whh[i].T @ dloss_dbfhthis
            dl_dwhy[i] += dloss_dwhythis
            dl_dwhh[i] += dloss_dwhhthis
        for i in range(ecdunits - 1, -1, -1):
            dloss_dbfhthis = dloss_dhlast * (1 - h[i + 1] ** 2)
            dloss_dwxhthis = dloss_dbfhthis @ x[i].T
            dloss_dwhhthis = dloss_dbfhthis @ h[i].T
            dloss_dhlast = whh[i].T @ dloss_dbfhthis
            dl_dwhh[i] += dloss_dwhhthis
            dl_dwxh[i] += dloss_dwxhthis
    loss_epo /= 700
    print('epoch:%7d,loss:%20.16f' %(epoch,loss_epo))
    if (epoch+1) % 50 == 0:
        va_loss = np.float64(0.0)
        for smp in range(24): # calculate loss of validation set
            x = np.reshape(x_va[smp], (ecdunits, 1, 1))
            h, y = RNN(whh, wxh, why, h, x, y, ecdunits, dcdunits)
            va_loss += np.mean((y - y_va_l[smp]) ** 2)
        va_loss /= 24
        print('epoch:',epoch,'oldvaloss:', valossold, 'newvaloss:', va_loss)
        if va_loss < valossold:
            np.save('wxh_h48.npy', wxh)
            np.save('whh_h48.npy', whh)
            np.save('why_h48.npy', why)
            print('va param saved.')
            valossold = va_loss
        elif loss_epo < lossold:
            np.save('wxhtr.npy', wxh)
            np.save('whhtr.npy', whh)
            np.save('whytr.npy', why)
            print('tr param saved.')
            lossold = loss_epo
    wxh = wxh - lr * dl_dwxh
    whh = whh - lr * dl_dwhh
    why = why - lr * dl_dwhy
#-------------------------- test --------------------------
fig, ax = plt.subplots(1, 4)
for smp in range(4):
    x = np.reshape(x_te[smp], (ecdunits, 1, 1))
    h, y = RNN(whh, wxh, why, h, x, y, ecdunits, dcdunits)
    delta_y = y - y_te_l[smp]
    print(y)
    print('test loss:',np.mean(delta_y ** 2))
    ax[smp].plot(np.arange(24),y_te_l[smp],'b-s')
    ax[smp].plot(np.arange(24),men+var*y,'r-s')
    ax[smp].grid(True)
plt.show()




# epoch:      0,loss: 14.5287313123181665
# epoch:      1,loss: 13.4270919367158967
# epoch:      2,loss: 10.1391590561648837
# epoch:      3,loss:  6.8080757714869744
# epoch:      4,loss:  6.2107768204353890
# epoch:      5,loss:  9.6282574561287486
# epoch:      6,loss:  3.0548050885314484
# epoch:      7,loss:  2.0986518296342447
# epoch:      8,loss:  1.7271847551247954
# epoch:      9,loss:  1.4852858441972239
# epoch:     10,loss:  1.2957008815142974
# epoch:     11,loss:  1.0621025518954219
# epoch:     12,loss:  1.1677163400709483
# epoch:     13,loss:  0.4170636386807888
# epoch:     14,loss:  0.2704879696715573
# epoch:     15,loss:  0.1879952019794904
# epoch:     16,loss:  0.1426816785850311
# epoch:     17,loss:  0.1249664915939114
# epoch:     18,loss:  0.1195401223299556
# epoch:     19,loss:  0.1170625605151376
# epoch:     20,loss:  0.1157084243284992
# epoch:     21,loss:  0.1145558291418173
# epoch:     22,loss:  0.1136051945927673
# epoch:     23,loss:  0.1126600705910990
# epoch:     24,loss:  0.1117850719048266
# epoch:     25,loss:  0.1108609866564861
# epoch:     26,loss:  0.1100078627107679
# epoch:     27,loss:  0.1091334662363776
# epoch:     28,loss:  0.1083024321071194
# epoch:     29,loss:  0.1074453839987269
# epoch:     30,loss:  0.1066304182214049
# epoch:     31,loss:  0.1057995344107472
# epoch:     32,loss:  0.1050003811642145
# epoch:     33,loss:  0.1041895120226493
# epoch:     34,loss:  0.1034044977979047
# epoch:     35,loss:  0.1026113600274714
# epoch:     36,loss:  0.1018372489289460
# epoch:     37,loss:  0.1010548369098328
# epoch:     38,loss:  0.1002789696482456
# epoch:     39,loss:  0.0994752121316745
# epoch:     40,loss:  0.0986132027444129
# epoch:     41,loss:  0.0976299348229009
# epoch:     42,loss:  0.0967658588555705
# epoch:     43,loss:  0.0960059353535554
# epoch:     44,loss:  0.0952367542625402
# epoch:     45,loss:  0.0944518897237018
# epoch:     46,loss:  0.0936676394592374
# epoch:     47,loss:  0.0928981460586041
# epoch:     48,loss:  0.0921537756857373
# epoch:     49,loss:  0.0914297588833674
# epoch:     50,loss:  0.0907218423204420
# epoch:     51,loss:  0.0900243062424222
# epoch:     52,loss:  0.0893344610504967
# epoch:     53,loss:  0.0886495536995013
# epoch:     54,loss:  0.0879684928866456
# epoch:     55,loss:  0.0872905102896473
# epoch:     56,loss:  0.0866159667467905
# epoch:     57,loss:  0.0859458326378395
# epoch:     58,loss:  0.0852818975257203
# epoch:     59,loss:  0.0846263324832776
# epoch:     60,loss:  0.0839813427714282
# epoch:     61,loss:  0.0833486399522775
# epoch:     62,loss:  0.0827291781185527
# epoch:     63,loss:  0.0821231351721730
# epoch:     64,loss:  0.0815301051238978
# epoch:     65,loss:  0.0809493402495314
# epoch:     66,loss:  0.0803799514288815
# epoch:     67,loss:  0.0798210399339943
# epoch:     68,loss:  0.0792717660056814
# epoch:     69,loss:  0.0787313781683931
# epoch:     70,loss:  0.0781992178737726
# epoch:     71,loss:  0.0776747135197583
# epoch:     72,loss:  0.0771573690587682
# epoch:     73,loss:  0.0766467517124428
# epoch:     74,loss:  0.0761424794886147
# epoch:     75,loss:  0.0756442092236146
# epoch:     76,loss:  0.0751516242176419
# epoch:     77,loss:  0.0746644202237861
# epoch:     78,loss:  0.0741822867830742
# epoch:     79,loss:  0.0737048782030337
# epoch:     80,loss:  0.0732317611000970
# epoch:     81,loss:  0.0727623056586202
# epoch:     82,loss:  0.0722954254421552
# epoch:     83,loss:  0.0718288494601716
# epoch:     84,loss:  0.0713568674828176
# epoch:     85,loss:  0.0708686892428423
# epoch:     86,loss:  0.0704000375444594
# epoch:     87,loss:  0.0699418143253571
# epoch:     88,loss:  0.0694885885580168
# epoch:     89,loss:  0.0690599335472398
# epoch:     90,loss:  0.0686382669344921
# epoch:     91,loss:  0.0682409190793739
# epoch:     92,loss:  0.0677459960771366
# epoch:     93,loss:  0.0674532056434162
# epoch:     94,loss:  0.0668952207095008
# epoch:     95,loss:  0.0664731154628899
# epoch:     96,loss:  0.0660789325529382
# epoch:     97,loss:  0.0656676749704842
# epoch:     98,loss:  0.0652862840612268
# epoch:     99,loss:  0.0648759339099695
# epoch:    100,loss:  0.0644918791798916
# epoch:    101,loss:  0.0641167861204821
# epoch:    102,loss:  0.0638671445697133
# epoch:    103,loss:  0.0633417482260806
# epoch:    104,loss:  0.0629474843781668
# epoch:    105,loss:  0.0625129863657354
# epoch:    106,loss:  0.0638131969853264
# epoch:    107,loss:  0.0611142975798170
# epoch:    108,loss:  0.0678268734226093
# epoch:    109,loss:  0.0646483897928954
# epoch:    110,loss:  0.0646917399534839
# epoch:    111,loss:  0.0591261245118590
# epoch:    112,loss:  0.0586843599827393
# epoch:    113,loss:  0.0577990868243777
# epoch:    114,loss:  0.0573470951679978
# epoch:    115,loss:  0.0568825657951306
# epoch:    116,loss:  0.0564625817455984
# epoch:    117,loss:  0.0560884108875911
# epoch:    118,loss:  0.0556537065002540
# epoch:    119,loss:  0.0552464159411322
# epoch:    120,loss:  0.0548539572280017
# epoch:    121,loss:  0.0544186117984940
# epoch:    122,loss:  0.0540335677267906
# epoch:    123,loss:  0.0536454887603371
# epoch:    124,loss:  0.0532377705735131
# epoch:    125,loss:  0.0528631817900726
# epoch:    126,loss:  0.0525053052583635
# epoch:    127,loss:  0.0570577162507607
# epoch:    128,loss:  0.0562888635986336
# epoch:    129,loss:  0.0554889942005865
# epoch:    130,loss:  0.0550676791500945
# epoch:    131,loss:  0.0546882664065468
# epoch:    132,loss:  0.0541986311408942
# epoch:    133,loss:  0.0537448232769571
# epoch:    134,loss:  0.0533196701834385
# epoch:    135,loss:  0.0528189376808955
# epoch:    136,loss:  0.0524276112873286
# epoch:    137,loss:  0.0519812730595713
# epoch:    138,loss:  0.0516312016119490
# epoch:    139,loss:  0.0513421261357555
# epoch:    140,loss:  0.0510941075629458
# epoch:    141,loss:  0.0507780893725254
# epoch:    142,loss:  0.0503939970100199
# epoch:    143,loss:  0.0501617100407186
# epoch:    144,loss:  0.0498400903647651
# epoch:    145,loss:  0.0496677301835364
# epoch:    146,loss:  0.0494776107619316
# epoch:    147,loss:  0.0492110870328911
# epoch:    148,loss:  0.0488671126578225
# epoch:    149,loss:  0.0486001848327516
# epoch:    150,loss:  0.0482248604422188
# epoch:    151,loss:  0.0478942183806365
# epoch:    152,loss:  0.0477284327452791
# epoch:    153,loss:  0.0476162572209622
# epoch:    154,loss:  0.0472940068258730
# epoch:    155,loss:  0.0470485527527181
# epoch:    156,loss:  0.0466991033469007
# epoch:    157,loss:  0.0464022704920112
# epoch:    158,loss:  0.0461097337963930
# epoch:    159,loss:  0.0459708507409771
# epoch:    160,loss:  0.0457866182551589
# epoch:    161,loss:  0.0455685640317322
# epoch:    162,loss:  0.0452389150318926
# epoch:    163,loss:  0.0449676358137920
# epoch:    164,loss:  0.0447127198687784
# epoch:    165,loss:  0.0444263234486189
# epoch:    166,loss:  0.0442410783835109
# epoch:    167,loss:  0.0441770482734902
# epoch:    168,loss:  0.0438762157176379
# epoch:    169,loss:  0.0436701733655427
# epoch:    170,loss:  0.0434130831403196
# epoch:    171,loss:  0.0431833267133201
# epoch:    172,loss:  0.0428956621215161
# epoch:    173,loss:  0.0426849940004838
# epoch:    174,loss:  0.0423357706726817
# epoch:    175,loss:  0.0424973331486423
# epoch:    176,loss:  0.0423085156923846
# epoch:    177,loss:  0.0421009316442922
# epoch:    178,loss:  0.0418384095437963
# epoch:    179,loss:  0.0416845015246569
# epoch:    180,loss:  0.0414462020066214
# epoch:    181,loss:  0.0412891696174331
# epoch:    182,loss:  0.0410535336271426
# epoch:    183,loss:  0.0407922816570351
# epoch:    184,loss:  0.0407518451862292
# epoch:    185,loss:  0.0405417152711905
# epoch:    186,loss:  0.0402119975684721
# epoch:    187,loss:  0.0401202660767484
# epoch:    188,loss:  0.0399762477567888
# epoch:    189,loss:  0.0395033672974843
# epoch:    190,loss:  0.0393700461860099
# epoch:    191,loss:  0.0391329783687465
# epoch:    192,loss:  0.0389945254474631
# epoch:    193,loss:  0.0387580866510849
# epoch:    194,loss:  0.0386328817963897
# epoch:    195,loss:  0.0383992237984544
# epoch:    196,loss:  0.0382379723181163
# epoch:    197,loss:  0.0381050941519260
# epoch:    198,loss:  0.0379081316802199
# epoch:    199,loss:  0.0377816412183575
# epoch:    200,loss:  0.0375812977639083
# epoch:    201,loss:  0.0374345463312227
# epoch:    202,loss:  0.0373341977618224
# epoch:    203,loss:  0.0371184093876538
# epoch:    204,loss:  0.0370370569187705
# epoch:    205,loss:  0.0368115926752738
# epoch:    206,loss:  0.0367353680489644
# epoch:    207,loss:  0.0365199435182514
# epoch:    208,loss:  0.0364105046702476
# epoch:    209,loss:  0.0362980536445147
# epoch:    210,loss:  0.0361322162147844
# epoch:    211,loss:  0.0359913563899761
# epoch:    212,loss:  0.0358259215539300
# epoch:    213,loss:  0.0356774053057866
# epoch:    214,loss:  0.0355551904554052
# epoch:    215,loss:  0.0353494909266142
# epoch:    216,loss:  0.0352640168668858
# epoch:    217,loss:  0.0350041485164501
# epoch:    218,loss:  0.0349148037566486
# epoch:    219,loss:  0.0346477911595573
# epoch:    220,loss:  0.0345474955084577
# epoch:    221,loss:  0.0342920980734471
# epoch:    222,loss:  0.0341877876501647
# epoch:    223,loss:  0.0339170289299794
# epoch:    224,loss:  0.0338007833573913
# epoch:    225,loss:  0.0335661433764122
# epoch:    226,loss:  0.0334565467814367
# epoch:    227,loss:  0.0332104812249920
# epoch:    228,loss:  0.0330861110926561
# epoch:    229,loss:  0.0328763079094197
# epoch:    230,loss:  0.0327094019932710
# epoch:    231,loss:  0.0325503697867660
# epoch:    232,loss:  0.0323366282509743
# epoch:    233,loss:  0.0322051089123418
# epoch:    234,loss:  0.0320599803670691
# epoch:    235,loss:  0.0318706727530366
# epoch:    236,loss:  0.0317751661850652
# epoch:    237,loss:  0.0315752398764890
# epoch:    238,loss:  0.0314886857772767
# epoch:    239,loss:  0.0313539434081433
# epoch:    240,loss:  0.0312473917249969
# epoch:    241,loss:  0.0310561478940045
# epoch:    242,loss:  0.0309417562626718
# epoch:    243,loss:  0.0307656056497264
# epoch:    244,loss:  0.0306106007911296
# epoch:    245,loss:  0.0305302102034912
# epoch:    246,loss:  0.0304002100252836
# epoch:    247,loss:  0.0302589225788531
# epoch:    248,loss:  0.0301323177379692
# epoch:    249,loss:  0.0300382200180781
# epoch:    250,loss:  0.0298698130000753
# epoch:    251,loss:  0.0297522954392592
# epoch:    252,loss:  0.0296088926310184
# epoch:    253,loss:  0.0294781580547821
# epoch:    254,loss:  0.0294145207912595
# epoch:    255,loss:  0.0293105188207367
# epoch:    256,loss:  0.0291540695054818
# epoch:    257,loss:  0.0290394471033673
# epoch:    258,loss:  0.0289410865243439
# epoch:    259,loss:  0.0288044339279181
# epoch:    260,loss:  0.0286481560536705
# epoch:    261,loss:  0.0285234882597093
# epoch:    262,loss:  0.0284460682782388
# epoch:    263,loss:  0.0283623212816184
# epoch:    264,loss:  0.0282230744515258
# epoch:    265,loss:  0.0281347873752979
# epoch:    266,loss:  0.0279796774494067
# epoch:    267,loss:  0.0278764368860094
# epoch:    268,loss:  0.0277704588467888
# epoch:    269,loss:  0.0276411108484599
# epoch:    270,loss:  0.0276391961976555
# epoch:    271,loss:  0.0274930116449889
# epoch:    272,loss:  0.0274138382869272
# epoch:    273,loss:  0.0272908970871615
# epoch:    274,loss:  0.0271555112955460
# epoch:    275,loss:  0.0270752271063567
# epoch:    276,loss:  0.0269689342039011
# epoch:    277,loss:  0.0268244978545143
# epoch:    278,loss:  0.0267397261824754
# epoch:    279,loss:  0.0266167035766141
# epoch:    280,loss:  0.0265611438607597
# epoch:    281,loss:  0.0265562306456922
# epoch:    282,loss:  0.0264532248628247
# epoch:    283,loss:  0.0263099949827075
# epoch:    284,loss:  0.0262274650454371
# epoch:    285,loss:  0.0261391388725440
# epoch:    286,loss:  0.0259981063847300
# epoch:    287,loss:  0.0259037786234428
# epoch:    288,loss:  0.0258177692493604
# epoch:    289,loss:  0.0257379270625532
# epoch:    290,loss:  0.0256022507969106
# epoch:    291,loss:  0.0255213681695872
# epoch:    292,loss:  0.0254076060935654
# epoch:    293,loss:  0.0253351650813149
# epoch:    294,loss:  0.0252460867742907
# epoch:    295,loss:  0.0251141040184136
# epoch:    296,loss:  0.0250330771621832
# epoch:    297,loss:  0.0249609230129347
# epoch:    298,loss:  0.0248282699888685
# epoch:    299,loss:  0.0247414839508201
# epoch: 49 oldvaloss: 1 newvaloss: 0.0010487487173195063
# va param saved.
# epoch: 99 oldvaloss: 0.0010487487173195063 newvaloss: 0.001043120472679214
# va param saved.
# epoch: 149 oldvaloss: 0.001043120472679214 newvaloss: 0.001035324497815244
# va param saved.
# epoch: 199 oldvaloss: 0.001035324497815244 newvaloss: 0.0012103687334793568
# tr param saved.
# epoch: 249 oldvaloss: 0.001035324497815244 newvaloss: 0.001042780978105939
# tr param saved.
# epoch: 299 oldvaloss: 0.001035324497815244 newvaloss: 0.0010249270541722736
# va param saved.