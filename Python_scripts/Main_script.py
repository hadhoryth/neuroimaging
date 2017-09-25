from data_learn import Analysis
from features_learn import Features
from Helper import Helpers


def run_analysis(data, keys, classifier='svm', scaler='minmax', printing=False,
                 resampling=False, pca_comp=None):
    anls, ft = Analysis(), Features()
    mixed = anls.mix_data(data, keys, printing)
    normilized = ft.scale_data(
        mixed['train']['features'], mixed['train']['labels'], scaler_type=scaler)

    if resampling:
        normilized['features'], normilized['labels'] = ft.resample_data(
            normilized['features'], normilized['labels'], r_type='tomek')

    if pca_comp is not None:
        normilized['features'], pca = ft.apply_pca(
            normilized['features'], pca_comp)

    acurracies, model_params = anls.train_model(classifier, normilized['features'],
                                                normilized['labels'], logging=printing)

    print('-----------Validating------------------')
    normilized_test = ft.scale_data(
        mixed['test']['features'], mixed['test']['labels'], scaler_type=scaler)
    ft.check_for_duplicates(normilized['features'], normilized_test['features'])

    if pca_comp is not None:
        normilized_test['features'] = pca.transform(normilized_test['features'])

    acurracies, model_params = anls.train_model(
        classifier, normilized_test['features'], normilized_test['labels'], params=model_params, logging=printing)


if __name__ == '__main__':
    print_logs = True
    info = {'mat_home': '/Users/XT/Documents/PhD/Granada/neuroimaging/ADNI_mat',
            'keys': ['normal', 'mci', 'ad', 'labels_normal', 'labels_mci', 'labels_ad']}
    hlp = Helpers()
    av45_data, fdg_data = hlp.get_features(
        info, '_all_data', logging=print_logs)

    # for i in range(40, 100):
    run_analysis(av45_data['av45'], info['keys'],
                 printing=print_logs, resampling=False, scaler='robust', pca_comp=90)
    # run_analysis(av45_data['av45'], info['keys'],
    #              printing=print_logs, resampling=True, scaler='robust',
    #              apply_pca=True, pca_comp=i)
    # run_analysis(fdg_data['fdg'], info['keys'],
    #              printing=print_logs, resampling=True)
