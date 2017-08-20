import numpy as np
from Helper import Helpers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os.path


class Features():
    def __init__(self):
        self.cerebellum_regions = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L',
                                   'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R',
                                   'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
                                   'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L',
                                   'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R',
                                   'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R', 'Cingulum_Ant_L',
                                   'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R', 'Hippocampus_L',
                                   'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R', 'Calcarine_L',
                                   'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R',
                                   'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R',
                                   'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R',
                                   'SupraMarginal_L', 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L',
                                   'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R',
                                   'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L',
                                   'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R',
                                   'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L',
                                   'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
                                   'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R',
                                   'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5',
                                   'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
        self.cerebellum_regions_lenght = len(self.cerebellum_regions)

    def _repackFeatures(self, data):
        brain_regions = dict()
        for i in range(len(data)):
            for j in range(self.cerebellum_regions_lenght):
                if self.cerebellum_regions[j] in brain_regions:
                    brain_regions[self.cerebellum_regions[j]] = np.append(
                        brain_regions[self.cerebellum_regions[j]], data[i][j])
                else:
                    brain_regions[self.cerebellum_regions[j]] = np.array(data[i][j])
        return brain_regions

    def _removeOutlayers(self, upd_data):
        scaler = MinMaxScaler()
        upd_data = scaler.fit_transform(upd_data.reshape(-1, 1))

        def replaceOutlayers(values, idx, std, max, min):
            for i in idx:
                if values[i] > max:
                    values[i] = max
                else:
                    values[i] = min
            return values
        mean, std = np.average(upd_data), 3 * np.std(upd_data)
        outlayers = np.append(np.where(upd_data < (mean - std))
                              [0], np.where(upd_data > (mean + std))[0])
        mask = np.ones(upd_data.shape, dtype=bool)
        mask[outlayers] = 0
        clean_max, clean_min = np.max(upd_data[mask]), np.min(upd_data[mask])

        upd_data = replaceOutlayers(upd_data, outlayers, std / 3, clean_max, clean_min)
        std = np.std(upd_data)
        if std < 0.1:
            return None
        return upd_data

    def _inverseRepackFeatures(self, data):
        """ data - dictionary """
        for value in data.values():
            if not 'out_data' in locals():
                out_data = [None] * len(value)
            for i in range(len(value)):
                if out_data[i] is None:
                    out_data[i] = np.array(value[i])
                else:
                    out_data[i] = np.append(out_data[i], value[i])

        return out_data

    def _drawFeatureStat(self, title, subplot_size, sub_data, count):
        def drawSubplot(data, ax, clr='g', ttl='Title'):
            def updateAxes(ax, title, xlabel, ylabel, mean, std):
                ax.set_title(title + ': ({0:0.2f}, '.format(mean) + '{0:0.2f})'.format(std))
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
            mean, std = np.mean(data), np.std(data)
            sns.distplot(data, color=clr, ax=ax)
            ax.plot((mean - 3 * std, mean - 3 * std), (0, 3))
            ax.plot((mean + 3 * std, mean + 3 * std), (0, 3))
            ax.plot((mean, mean), (0, 3))
            updateAxes(ax, ttl, 'Frequency', 'Density function value', mean, std)

        rows, columns = subplot_size[0], subplot_size[1]
        f, axes = plt.subplots(rows, columns, figsize=(9, 7), sharex=True)
        sns.despine(left=True)
        f.suptitle(title, fontweight='bold')
        k = 0
        patient_type = ['Normal', 'MCI', 'AD']
        clr = ['g', 'm', 'b']
        for row in range(rows):
            for clm in range(columns):
                drawSubplot(sub_data[k], axes[row, clm], clr[clm], patient_type[clm])
                k += 1

        plt.savefig(os.path.join('_features', str(count) + '_' + title))
        plt.close()
        # plt.show()

    def normalizeFeatures(self, brain_data, saveFeatures=False):
        brain_normal = self._repackFeatures(brain_data['normal'])
        brain_mci = self._repackFeatures(brain_data['mci'])
        brain_ad = self._repackFeatures(brain_data['ad'])

        new_brain_normal, new_brain_mci, new_brain_ad = dict(), dict(), dict()
        num = 1
        for region in self.cerebellum_regions:
            norm = self._removeOutlayers(np.copy(brain_normal[region]))
            mci = self._removeOutlayers(np.copy(brain_mci[region]))
            ad = self._removeOutlayers(np.copy(brain_ad[region]))

            if norm is None or (mci is None) or (ad is None):
                num += 1
                continue

            new_brain_normal[region], new_brain_mci[region], new_brain_ad[region] = norm, mci, ad

            if saveFeatures:
                toDraw = [brain_normal[region], brain_mci[region],
                          brain_ad[region], norm, mci, ad]
                self._drawFeatureStat(region, (2, 3), toDraw, num)
            num += 1

        brain_regions_norm = self._inverseRepackFeatures(new_brain_normal)
        brain_regions_mci = self._inverseRepackFeatures(new_brain_mci)
        brain_regions_ad = self._inverseRepackFeatures(new_brain_ad)

        return {'normal': brain_regions_norm, 'mci': brain_regions_mci, 'ad': brain_regions_ad,
                'labels_normal': np.zeros(len(brain_regions_norm)),
                'labels_mci': np.zeros(len(brain_regions_mci)) + 1,
                'labels_ad': np.zeros(len(brain_regions_ad)) + 2}
