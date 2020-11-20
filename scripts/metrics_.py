import numpy as np
from scipy.stats import pearsonr
import math

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
# from dtw import dtw


def cohen_kappa(x1, x2):

	return cohen_kappa_score(x1, x2)


def ccc(original, prediction):
    true_mean = np.mean(original)
    true_variance = np.var(original)
    pred_mean = np.mean(prediction)
    pred_variance = np.var(prediction)

    rho, _ = pearsonr(prediction,original)

    if math.isnan(rho):
        rho = 0

    std_predictions = np.std(prediction)
    std_gt = np.std(original)

    ccc_score = 2 * rho * std_gt * std_predictions / (std_predictions ** 2 + std_gt ** 2 + (pred_mean - true_mean) ** 2)

    return ccc_score, rho 


def pcc(orig, pred):
	pcc_score, p_value = pearsonr(orig, pred)

	return pcc_score, p_value


def mcc(orig, pred):

	return matthews_corrcoef(orig, pred)


def mse(orig, pred):
	return mean_squared_error(orig, pred) 


def rmse(orig, pred):
	return np.sqrt(mse(orig, pred))


def mae(orig, pred):
	return mean_absolute_error(orig, pred)
	

# def dt_warping(orig, pred):
# 	# reshape data 
# 	orig = orig.reshape(-1, 1)
# 	pred = pred.reshape(-1, 1)

# 	euclidean_norm = lambda orig, pred: np.abs(orig - pred)

# 	d_score, cost_matrix, acc_cost_matrix, path = dtw(orig, pred, dist=euclidean_norm)

# 	return d_score, path


def eval_regression(orig, pred):
	mae_score = mae(orig, pred)
	mse_score = mse(orig, pred)
	rmse_score = rmse(orig, pred)
	pcc_score, p_value = pcc(orig, pred)
	ccc_score, _ = ccc(orig, pred)
	# dtw_score, _ = dt_warping(orig, pred)

	eval_names = ["MAE", "MSE", "RMSE", "PCC", "CCC", "DT Warping"]
	eval_scores = [("MAE", round(mae_score, 6)), 
					("MSE", round(mse_score, 6)), 
					("RMSE", round(rmse_score, 6)), 
					("PCC", round(pcc_score, 6)), 
					("CCC", round(ccc_score, 6)), 
					# ("DT Warping", round(dtw_score, 6))
					]

	return eval_scores


def accuracy(orig, pred):
	return accuracy_score(orig, pred)


def precision(orig, pred):
	return precision_score(orig, pred)


def recall(orig, pred):
	return recall_score(orig, pred)


def f1score(orig, pred):
	return f1_score(orig, pred)


def f1_s_macro(orig, pred):
	return f1_score(orig, pred, average='macro')


def f1_s_micro(orig, pred):
	return f1_score(orig, pred, average="micro")


def auc_roc(orig, pred_prob):
	return roc_auc_score(orig, pred_prob)


def eval_classification(orig, pred, pred_prob):
	acc_score = accuracy(orig, pred)
	pre_score = precision(orig, pred)
	rec_score = recall(orig, pred)
	f_score = f1score(orig, pred)
	auc_score = auc_roc(orig, pred_prob)

	eval_scores = [("Accuracy", round(acc_score, 4)),
					("Precision", round(pre_score, 4)),
					("Recall", round(rec_score, 4)),
					("F1_score", round(f_score, 4)),
					("AUC_ROC_score", round(auc_score, 4))
					]

	return eval_scores


if __name__ == "__main__":
	orig = np.array([1, 2, 3, 4])
	pred = np.array([0.8, 1, 3.2, 4.5])

	ccc_score, _ = ccc(orig, pred)
	print("CCC score:", ccc_score)

	pcc_score, p_value = pcc(orig, pred)
	print("PCC score:", pcc_score, ", P value:", p_value)

	mse_score = mse(orig, pred)
	print("MSE score:", mse_score)

	rmse_score = rmse(orig, pred)
	print("RMSE score:", rmse_score)

	mae_score = mae(orig, pred)
	print("MAE score:", mae_score)

	d_score, _ = dt_warping(orig, pred)
	print("Dynamic time warping:", d_score)

	eval_scores = eval_regression(orig, pred)
	print("Evaluation results:", eval_scores)

	




