import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    
    ppm_accs = [67.49594211578369,66.39813780784607,65.09863535563152,65.39833545684814,63.19852868715922,58.1015149752299,57.60011672973633]
    ppm_stds = [2.8819731108612365,1.3189940535181308,1.293359505706777, 1.2043492455765807,1.1506803821974771,1.4522614481507272,0.984146318014543]
    pd_accs = [67.29963620503743,66.20063384373982,64.3979291121165,63.50122690200806,60.10051965713501,55.50160805384318,53.90180548032125]
    pd_stds = [0.28575375333610836,0.7584965576671632,1.9088382821347614,2.032984953414279,1.9956756017534307,2.1626005423134194,2.6564388031853783]
    outlier_percts = list(range(len(ppm_accs)))
    plt.figure(figsize=(10,6))
    plt.plot(outlier_percts,pd_accs,marker='o')
    plt.plot(outlier_percts,ppm_accs,marker='o')
    plt.legend(['PD-FL','PPM-FL'],fontsize=20,loc='lower left')
    plt.xlabel(r'$\epsilon$',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.yticks(fontsize=20)
    plt.yticks([50,55,60,65,70],[50,55,60,65,70])
    plt.ylim(50,70)
    plt.xticks(fontsize=20)
    plt.xticks(outlier_percts,
               [r'$0.0\%$',r'$2.5\%$',r'$5.0\%$',r'$7.5\%$',r'$10.0\%$',r'$12.5\%$',r'$15.0\%$'])
    plt.savefig('robustness_clf.png')
    plt.close()