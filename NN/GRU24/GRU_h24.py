import numpy as np
import matplotlib.pyplot as plt
ecdunits, dcdunits, hlen = 48, 24, 24
tb = 0 # test bias
lr = 0.25  # learning rate
'''
real hour-price prediction based on GRU
作者：陈长 
20210109
'''
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def getxy(data, start, end, ecdunits, dcdunits):
    x, y = np.array([0.0]), np.array([0.0])
    for i in range(start, end):
        x = np.append(x, data[i])
        x = np.append(x, data[i + 1])
        y = np.append(y, data[i + 2])
    x, y = np.delete(x, 0), np.delete(y, 0)
    return np.reshape(x, (-1, ecdunits)), np.reshape(y, (-1, dcdunits))

def GRU_v2(Wr, Ur, Wz, Uz, Wh, Uh, V, h, x, y, r, z, ecdunits, dcdunits):
    for i in range(ecdunits):  # encode
        r[i], z[i] = sig(Wr[i] @ h[i] + Ur[i] @ x[i]), sig(Wz[i] @ h[i] + Uz[i] @ x[i])
        hpp[i] = r[i] * h[i]
        hp[i] = np.tanh(Wh[i] @ hpp[i] + Uh[i] @ x[i])
        h[i + 1] = (1 - z[i]) * h[i] + z[i] * hp[i]
    for i in range(ecdunits, ecdunits + dcdunits):  # decode
        r[i], z[i] = sig(Wr[i] @ h[i]), sig(Wz[i] @ h[i])
        hpp[i] = r[i] * h[i]
        hp[i] = np.tanh(Wh[i] @ hpp[i])
        h[i + 1] = (1 - z[i]) * h[i] + z[i] * hp[i]
        y[i - ecdunits] = V[i - ecdunits] @ h[i + 1]
    return r, z, hpp, hp, h, y
# -------------------------- load data --------------------------
data = np.loadtxt(r'price_730ds.csv')  # shape = (17520,)
data = np.reshape(data, (-1, dcdunits))  # shape = (730, 24)
x_tr, y_tr_l = getxy(data, 0, 700, ecdunits, dcdunits)
x_va, y_va_l = getxy(data, 700, 724, ecdunits, dcdunits)
x_te, y_te_l = getxy(data, 724 - tb, 728 - tb, ecdunits, dcdunits)
men, var = np.mean(x_tr), np.var(x_tr)
x_tr, y_tr_l, x_va, y_va_l, x_te = (x_tr - men) / var, (y_tr_l - men) / var, (x_va - men) / var, (y_va_l - men) / var, (x_te - men) / var
Wr = np.tanh( np.random.normal(size=(ecdunits + dcdunits, hlen, hlen)) )
Wz = np.tanh( np.random.normal(size=(ecdunits + dcdunits, hlen, hlen)) )
Wh = np.tanh( np.random.normal(size=(ecdunits + dcdunits, hlen, hlen)) )
Ur = np.tanh( np.random.normal(size=(ecdunits, hlen, 1)) )
Uz = np.tanh( np.random.normal(size=(ecdunits, hlen, 1)) )
Uh = np.tanh( np.random.normal(size=(ecdunits, hlen, 1)) )
V = np.tanh( np.random.normal(size=(dcdunits, 1, hlen)) )
# Wr = np.load('Wr_tr.npy')
# Wz = np.load('Wz_tr.npy')
# Wh = np.load('Wh_tr.npy')
# Ur = np.load('Ur_tr.npy')
# Uz = np.load('Uz_tr.npy')
# Uh = np.load('Uh_tr.npy')
# V = np.load('V_tr.npy')
h = np.zeros((ecdunits + dcdunits + 1, hlen, 1))  # state vector h initialization
hpp, hp = np.zeros((ecdunits + dcdunits, hlen, 1)), np.zeros(
    (ecdunits + dcdunits, hlen, 1))  # state vector h initialization
r, z = np.zeros((ecdunits + dcdunits, hlen, 1)), np.zeros(
    (ecdunits + dcdunits, hlen, 1))  # state vector h initialization
y = np.zeros(dcdunits)  # output vector y initialization
# -------------------------- training process --------------------------
lossold = 1.0e6  # old loss initialization
valossold = 1
for epoch in range(np.int64(3e9)):  # train the whole tr set once
    dl_dWr, dl_dWz, dl_dWh, dl_dUr, dl_dUz, dl_dUh, dl_dV = np.zeros_like(Wr), np.zeros_like(Wz), np.zeros_like(Wh), np.zeros_like(Ur), np.zeros_like(Uz), np.zeros_like(Uh), np.zeros_like(V)
    loss_epo = np.float64(0.0)
    for smp in range(700):
        x = np.reshape(x_tr[smp], (ecdunits, 1, 1))
        r, z, hpp, hp, h, y = GRU_v2(Wr, Ur, Wz, Uz, Wh, Uh, V, h, x, y, r, z, ecdunits, dcdunits)
        delta_y = y - y_tr_l[smp]
        loss_epo += np.mean(delta_y ** 2)
        dloss_dhlast = 0
        for i in range(-1, -(dcdunits + 1), -1):
            dl_dV[i] += 2. / dcdunits * (delta_y[i]) * h[i].T
            dloss_dhthis = 2. / dcdunits * (delta_y[i]) * V[i].T + dloss_dhlast
            dl_dWr[i] += (Wh[i].T @ (dloss_dhthis * z[i] * (1 - hp[i] ** 2)) * h[i - 1] * r[i] * (1 - r[i])) @ h[i - 1].T
            dl_dWh[i] += (dloss_dhthis * z[i] * (1 - hp[i] ** 2)) @ hpp[i].T
            dl_dWz[i] += (dloss_dhthis * (hp[i] - h[i - 1]) * z[i] * (1 - z[i])) @ h[i - 1].T
            v = Wh[i].T @ (dloss_dhthis * z[i] * (1 - hp[i] ** 2))
            dloss_dhlast = v * r[i] + Wr[i].T @ (v * h[i - 1] * r[i] * (1 - r[i])) + dloss_dhthis * (1 - z[i]) + Wz[i].T @ (dloss_dhthis * (hp[i] - h[i - 1]) * z[i] * (1 - z[i]))
        for i in range(ecdunits - 1, -1, -1):
            dl_dWr[i] += (Wh[i].T @ (dloss_dhlast * z[i] * (1 - hp[i] ** 2)) * h[i] * r[i] * (1 - r[i])) @ h[i].T
            dl_dUr[i] += (Wh[i].T @ (dloss_dhlast * z[i] * (1 - hp[i] ** 2)) * h[i] * r[i] * (1 - r[i])) @ x[i].T
            dl_dWh[i] += (dloss_dhlast * z[i] * (1 - hp[i] ** 2)) @ hpp[i].T
            dl_dUh[i] += (dloss_dhlast * z[i] * (1 - hp[i] ** 2)) @ x[i].T
            dl_dWz[i] += (dloss_dhlast * (hp[i] - h[i]) * z[i] * (1 - z[i])) @ h[i].T
            dl_dUz[i] += (dloss_dhlast * (hp[i] - h[i]) * z[i] * (1 - z[i])) @ x[i].T
            v = Wh[i].T @ (dloss_dhlast * z[i] * (1 - hp[i] ** 2))
            dloss_dhlast = v * r[i] + Wr[i].T @ (v * h[i] * r[i] * (1 - r[i])) + dloss_dhlast * (1 - z[i]) + Wz[i].T @ (dloss_dhlast * (hp[i] - h[i]) * z[i] * (1 - z[i]))
    loss_epo /= 700
    print('epoch:%7d,loss:%20.16f' %(epoch,loss_epo))
    if (epoch + 1) % 50 == 0:
        va_loss = np.float64(0.0)
        for smp in range(24):  # calculate loss of validation set
            x = np.reshape(x_va[smp], (ecdunits, 1, 1))
            r, z, hpp, hp, h, y = GRU_v2(Wr, Ur, Wz, Uz, Wh, Uh, V, h, x, y, r, z, ecdunits, dcdunits)
            va_loss += np.mean((y - y_va_l[smp]) ** 2)
        va_loss /= 24
        print('epoch:', epoch, 'oldvaloss:', valossold, 'newvaloss:', va_loss)
        if va_loss < valossold:
            np.save('Wr.npy', Wr), np.save('Wh.npy', Wh), np.save('Wz.npy', Wz), np.save('Ur.npy', Ur), np.save('Uh.npy', Uh), np.save('Uz.npy', Uz), np.save('V.npy', V)
            print('va param saved.')
            valossold = va_loss
        elif loss_epo < lossold:
            np.save('Wr_tr.npy', Wr), np.save('Wh_tr.npy', Wh), np.save('Wz_tr.npy', Wz), np.save('Ur_tr.npy', Ur), np.save('Uh_tr.npy', Uh), np.save('Uz_tr.npy', Uz), np.save('V_tr.npy', V)
            print('tr param saved.')
            lossold = loss_epo
    Wr -= lr * dl_dWr
    Wh -= lr * dl_dWh
    Wz -= lr * dl_dWz
    Ur -= lr * dl_dUr
    Uh -= lr * dl_dUh
    Uz -= lr * dl_dUz
    V -= lr * dl_dV
# -------------------------- test --------------------------
fig, ax = plt.subplots(1, 4)
for smp in range(4):
    x = np.reshape(x_te[smp], (ecdunits, 1, 1))
    r, z, hpp, hp, h, y = GRU_v2(Wr, Ur, Wz, Uz, Wh, Uh, V, h, x, y, r, z, ecdunits, dcdunits)
    delta_y = y - y_te_l[smp]
    print(y)
    print('test loss:', np.mean(delta_y ** 2))
    ax[smp].plot(np.arange(24), y_te_l[smp], 'b-s')
    ax[smp].plot(np.arange(24), men + var * y, 'r-s')
    ax[smp].grid(True)
plt.show()

# -------------------------- training process record --------------------------
# epoch:      0,loss:  0.0030671926438552
# epoch:      1,loss:  0.0023895849267623
# epoch:      2,loss:  0.0021776657729515
# epoch:      3,loss:  0.0020482550681531
# epoch:      4,loss:  0.0019711091539397
# epoch:      5,loss:  0.0019180123091897
# epoch:      6,loss:  0.0018768508579387
# epoch:      7,loss:  0.0018445298914892
# epoch:      8,loss:  0.0018165027946601
# epoch:      9,loss:  0.0017955536867152
# epoch:     10,loss:  0.0017700591900641
# epoch:     11,loss:  0.0017519649018336
# epoch:     12,loss:  0.0017299794921713
# epoch:     13,loss:  0.0017135237392621
# epoch:     14,loss:  0.0016977236085870
# epoch:     15,loss:  0.0016842397354293
# epoch:     16,loss:  0.0016717100718557
# epoch:     17,loss:  0.0016607852980300
# epoch:     18,loss:  0.0016505355350142
# epoch:     19,loss:  0.0016419783020541
# epoch:     20,loss:  0.0016334190602405
# epoch:     21,loss:  0.0016278580754014
# epoch:     22,loss:  0.0016191521867317
# epoch:     23,loss:  0.0016160157325956
# epoch:     24,loss:  0.0016046925353632
# epoch:     25,loss:  0.0016008994449055
# epoch:     26,loss:  0.0015901262041959
# epoch:     27,loss:  0.0015857651974329
# epoch:     28,loss:  0.0015760343422055
# epoch:     29,loss:  0.0015715395766909
# epoch:     30,loss:  0.0015626536934140
# epoch:     31,loss:  0.0015583172694415
# epoch:     32,loss:  0.0015499127264830
# epoch:     33,loss:  0.0015458563817491
# epoch:     34,loss:  0.0015376457647147
# epoch:     35,loss:  0.0015338779889193
# epoch:     36,loss:  0.0015256817523383
# epoch:     37,loss:  0.0015221240955611
# epoch:     38,loss:  0.0015138847458821
# epoch:     39,loss:  0.0015104183424978
# epoch:     40,loss:  0.0015021847372736
# epoch:     41,loss:  0.0014987108474321
# epoch:     42,loss:  0.0014905776941013
# epoch:     43,loss:  0.0014870492016391
# epoch:     44,loss:  0.0014790953901279
# epoch:     45,loss:  0.0014755100247432
# epoch:     46,loss:  0.0014677745373972
# epoch:     47,loss:  0.0014641537118624
# epoch:     48,loss:  0.0014566439613990
# epoch:     49,loss:  0.0014530163387880
# epoch:     50,loss:  0.0014457266851430
# epoch:     51,loss:  0.0014421202538376
# epoch:     52,loss:  0.0014350466684147
# epoch:     53,loss:  0.0014314861818974
# epoch:     54,loss:  0.0014246336807726
# epoch:     55,loss:  0.0014211402602174
# epoch:     56,loss:  0.0014145245239527
# epoch:     57,loss:  0.0014111158385257
# epoch:     58,loss:  0.0014047612045612
# epoch:     59,loss:  0.0014014517429789
# epoch:     60,loss:  0.0013953872610945
# epoch:     61,loss:  0.0013921886198780
# epoch:     62,loss:  0.0013864433745693
# epoch:     63,loss:  0.0013833645657740
# epoch:     64,loss:  0.0013779632181458
# epoch:     65,loss:  0.0013750109967268
# epoch:     66,loss:  0.0013699703080132
# epoch:     67,loss:  0.0013671495016846
# epoch:     68,loss:  0.0013624763172692
# epoch:     69,loss:  0.0013597901085662
# epoch:     70,loss:  0.0013554809212361
# epoch:     71,loss:  0.0013529310142809
# epoch:     72,loss:  0.0013489728973553
# epoch:     73,loss:  0.0013465595396069
# epoch:     74,loss:  0.0013429320292968
# epoch:     75,loss:  0.0013406539355110
# epoch:     76,loss:  0.0013373313697175
# epoch:     77,loss:  0.0013351856508689
# epoch:     78,loss:  0.0013321395183618
# epoch:     79,loss:  0.0013301217114699
# epoch:     80,loss:  0.0013273226901082
# epoch:     81,loss:  0.0013254269288271
# epoch:     82,loss:  0.0013228464446648
# epoch:     83,loss:  0.0013210657484350
# epoch:     84,loss:  0.0013186770240644
# epoch:     85,loss:  0.0013170036484676
# epoch:     86,loss:  0.0013147823013705
# epoch:     87,loss:  0.0013132080897758
# epoch:     88,loss:  0.0013111323845982
# epoch:     89,loss:  0.0013096490786912
# epoch:     90,loss:  0.0013076999420706
# epoch:     91,loss:  0.0013062994310319
# epoch:     92,loss:  0.0013044603204107
# epoch:     93,loss:  0.0013031348258161
# epoch:     94,loss:  0.0013013915189659
# epoch:     95,loss:  0.0013001337221077
# epoch:     96,loss:  0.0012984740709937
# epoch:     97,loss:  0.0012972771925946
# epoch:     98,loss:  0.0012956908677791
# epoch:     99,loss:  0.0012945487096503
# epoch:    100,loss:  0.0012930269502544
# epoch:    101,loss:  0.0012919339067333
# epoch:    102,loss:  0.0012904692847482
# epoch:    103,loss:  0.0012894203301986
# epoch:    104,loss:  0.0012880065347717
# epoch:    105,loss:  0.0012869971925684
# epoch:    106,loss:  0.0012856288380988
# epoch:    107,loss:  0.0012846551362308
# epoch:    108,loss:  0.0012833275966141
# epoch:    109,loss:  0.0012823860149426
# epoch:    110,loss:  0.0012810952846847
# epoch:    111,loss:  0.0012801826986256
# epoch:    112,loss:  0.0012789252798688
# epoch:    113,loss:  0.0012780389046724
# epoch:    114,loss:  0.0012768117176959
# epoch:    115,loss:  0.0012759490565893
# epoch:    116,loss:  0.0012747493703120
# epoch:    117,loss:  0.0012739081686944
# epoch:    118,loss:  0.0012727335472243
# epoch:    119,loss:  0.0012719117540381
# epoch:    120,loss:  0.0012707600153428
# epoch:    121,loss:  0.0012699557518154
# epoch:    122,loss:  0.0012688249350161
# epoch:    123,loss:  0.0012680364702616
# epoch:    124,loss:  0.0012669248087071
# epoch:    125,loss:  0.0012661505412234
# epoch:    126,loss:  0.0012650564392440
# epoch:    127,loss:  0.0012642948831180
# epoch:    128,loss:  0.0012632168950717
# epoch:    129,loss:  0.0012624666696835
# epoch:    130,loss:  0.0012614034805029
# epoch:    131,loss:  0.0012606633026431
# epoch:    132,loss:  0.0012596137095417
# epoch:    133,loss:  0.0012588823870643
# epoch:    134,loss:  0.0012578452823450
# epoch:    135,loss:  0.0012571217087278
# epoch:    136,loss:  0.0012560960637706
# epoch:    137,loss:  0.0012553792131965
# epoch:    138,loss:  0.0012543640637261
# epoch:    139,loss:  0.0012536529865018
# epoch:    140,loss:  0.0012526474191789
# epoch:    141,loss:  0.0012519412374662
# epoch:    142,loss:  0.0012509443777553
# epoch:    143,loss:  0.0012502422816911
# epoch:    144,loss:  0.0012492532828608
# epoch:    145,loss:  0.0012485545272055
# epoch:    146,loss:  0.0012475725602373
# epoch:    147,loss:  0.0012468764617224
# epoch:    148,loss:  0.0012459007058488
# epoch:    149,loss:  0.0012452066414155
# epoch:    150,loss:  0.0012442362749837
# epoch:    151,loss:  0.0012435436811276
# epoch:    152,loss:  0.0012425778724824
# epoch:    153,loss:  0.0012418862459568
# epoch:    154,loss:  0.0012409241440553
# epoch:    155,loss:  0.0012402330442574
# epoch:    156,loss:  0.0012392737687655
# epoch:    157,loss:  0.0012385828222308
# epoch:    158,loss:  0.0012376254529149
# epoch:    159,loss:  0.0012369343605044
# epoch:    160,loss:  0.0012359779258332
# epoch:    161,loss:  0.0012352864734037
# epoch:    162,loss:  0.0012343299384517
# epoch:    163,loss:  0.0012336380120683
# epoch:    164,loss:  0.0012326802660970
# epoch:    165,loss:  0.0012319878731421
# epoch:    166,loss:  0.0012310277176792
# epoch:    167,loss:  0.0012303350154278
# epoch:    168,loss:  0.0012293711542450
# epoch:    169,loss:  0.0012286784873003
# epoch:    170,loss:  0.0012277095201946
# epoch:    171,loss:  0.0012270174667487
# epoch:    172,loss:  0.0012260418886944
# epoch:    173,loss:  0.0012253513108220
# epoch:    174,loss:  0.0012243675149353
# epoch:    175,loss:  0.0012236795954006
# epoch:    176,loss:  0.0012226858685545
# epoch:    177,loss:  0.0012220020870758
# epoch:    178,loss:  0.0012209965671781
# epoch:    179,loss:  0.0012203185110352
# epoch:    180,loss:  0.0012192990522618
# epoch:    181,loss:  0.0012186278716702
# epoch:    182,loss:  0.0012175917887096
# epoch:    183,loss:  0.0012169270698478
# epoch:    184,loss:  0.0012158709309609
# epoch:    185,loss:  0.0012152090009519
# epoch:    186,loss:  0.0012141290887076
# epoch:    187,loss:  0.0012134615669222
# epoch:    188,loss:  0.0012123558256499
# epoch:    189,loss:  0.0012116701736248
# epoch:    190,loss:  0.0012105411649700
# epoch:    191,loss:  0.0012098243170573
# epoch:    192,loss:  0.0012086802886165
# epoch:    193,loss:  0.0012079234633013
# epoch:    194,loss:  0.0012067751856425
# epoch:    195,loss:  0.0012059762442282
# epoch:    196,loss:  0.0012048318701601
# epoch:    197,loss:  0.0012039941056462
# epoch:    198,loss:  0.0012028563279209
# epoch:    199,loss:  0.0012019857896909

# epoch: 49 oldvaloss: 1 newvaloss: 0.000746184424541714
# va param saved.
# epoch: 99 oldvaloss: 0.000746184424541714 newvaloss: 0.000674166563471665
# va param saved.
# epoch: 149 oldvaloss: 0.000674166563471665 newvaloss: 0.0006843407091767454
# tr param saved.
# epoch: 199 oldvaloss: 0.000674166563471665 newvaloss: 0.0007085774625743792
# tr param saved.