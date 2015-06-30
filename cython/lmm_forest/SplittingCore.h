/*
 * SplittingCore.h
 *
 *  Created on: Jan 25, 2012
 *      Author: stegle
 *
 *
 */

#ifndef SPLITTINGCORE_H_
#define SPLITTINGCORE_H_
#include "types.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include <time.h>


typedef std::pair<mint_t, mfloat_t> argsort_pair;

inline bool argsort_comp(const argsort_pair& left, const argsort_pair& right) {
    return left.second < right.second;
}

template<typename Derived1,typename Derived2, typename Derived3>
inline void indsort(const Eigen::MatrixBase<Derived1> &iin, const Eigen::MatrixBase<Derived2> &iout_, const Eigen::MatrixBase<Derived3> &x)
{
	Eigen::MatrixBase<Derived2>& iout = const_cast< Eigen::MatrixBase<Derived2>& >(iout_);
	iout.derived().resize(x.size(),1);
    std::vector<argsort_pair> data(x.size());
    for(muint_t i=0;i<(muint_t)x.size();++i) {
        data[i].first = iin(i);
        data[i].second = x(i);
    }
    std::sort(data.begin(), data.end(), argsort_comp);
    for(muint_t i=0;i<(muint_t)data.size();++i) {
    	iout(i) = data[i].first;
    }
}

inline MatrixXi range(muint_t size){
	MatrixXi out(size,1);
	for(muint_t i=0;i<(muint_t)size; ++i){
		out(i,0) = i;
	}
	return out;
}

template<typename Derived1,typename Derived2>
inline void argsort(const Eigen::MatrixBase<Derived1> &iout_, const Eigen::MatrixBase<Derived2> &x)
{
	Eigen::MatrixBase<Derived1>& iout = const_cast< Eigen::MatrixBase<Derived1>& >(iout_);
	iout.derived().resize(x.size());
    //
    std::vector<argsort_pair> data(x.size());
    for(muint_t i=0;i<(muint_t)x.size();++i) {
        data[i].first = i;
        data[i].second = x(i);
    }
    std::sort(data.begin(), data.end(), argsort_comp);
    for(muint_t i=0;i<(muint_t)data.size();++i) {
    	iout(data[i].first) = i;
    }
}

template<typename Derived1,typename Derived2>
inline MatrixXd eigenIndex(const Eigen::MatrixBase<Derived1>& mout_,const Eigen::MatrixBase<Derived2>& m,const VectorXi& ind)
{
	Eigen::MatrixBase<Derived1>& mout = const_cast< Eigen::MatrixBase<Derived1>& >(mout_);
	//create result matrix
	mout.derived().resize(ind.rows(),m.cols());
	for(muint_t i=0;i<(muint_t)ind.rows();i++)
		mout.row(i) = m.row(ind(i));
	return mout;
}

template<typename Derived1,typename Derived2>
inline MatrixXd eigenColumn(const Eigen::MatrixBase<Derived1>& mout_,const Eigen::MatrixBase<Derived2>& m,const VectorXi& ind)
{
	Eigen::MatrixBase<Derived1>& mout = const_cast< Eigen::MatrixBase<Derived1>& >(mout_);
	//create result matrix
	for(muint_t i=0;i<(muint_t)ind.rows();i++)
		mout.col(i) = m.col(ind(i));
	return mout;
}

template<typename Derived1,typename Derived2>
inline void eigen_submatrix(const Eigen::MatrixBase<Derived1>& mout_,const Eigen::MatrixBase<Derived2>& m,const VectorXi& row_ind, const VectorXi& col_ind){
	MatrixXd m_tmp = eigenIndex(mout_, m, row_ind);
	MatrixXd mout(row_ind.rows(), col_ind.rows());
	m_tmp.transposeInPlace();
	mout = eigenIndex(mout, m_tmp, col_ind);
	mout.transposeInPlace();
}


template<typename Derived1,typename Derived2>
inline mfloat_t indSum(const Eigen::MatrixBase<Derived1>& m, const Eigen::MatrixBase<Derived2>& ind)
{
	mfloat_t sum = 0.0;
	for(muint_t i=0;i<(muint_t)ind.rows();i++){
			sum += m(ind(i));
	}
	return sum;
}

template <typename Derived1,typename Derived2>
inline void copy(const Eigen::MatrixBase<Derived1>* out_, const Eigen::MatrixBase<Derived2>* in_)
{
	const_cast< Eigen::MatrixBase<Derived1>& >(*out_) = (*in_);
}

template <typename Derived1,typename Derived2>
inline MatrixXd dot_by_index(
		const Eigen::MatrixBase<Derived1>* m1,
		const Eigen::MatrixBase<Derived1>* m2,
		const Eigen::MatrixBase<Derived2>* m1_rows,
		const Eigen::MatrixBase<Derived2>* m1_cols,
		const Eigen::MatrixBase<Derived2>* m2_cols)
{
	muint_t n = m1_rows->rows();
	muint_t m = m2_cols->rows();
	MatrixXd prod(n,m);

	for (muint_t i=0; i<n; ++i){
		for (muint_t j=0; j<m; ++j){
			prod(i,j) = 0;
			for (muint_t k=0; k<((muint_t)m1_cols->rows()); ++k){
				prod(i,j) += (*m1)((*m1_rows)(i), (*m1_cols)(k))*(*m2)((*m1_cols)(k), (*m2_cols)(j));
			}
		}
	}
	return prod;
}

template <typename Derived1,typename Derived2, typename Derived3>
inline void ml_beta(const Eigen::MatrixBase<Derived1>* out_,
		const Eigen::MatrixBase<Derived2>* UTX,
		const Eigen::MatrixBase<Derived3>* UTY,
		const Eigen::MatrixBase<Derived3>* S, mfloat_t delta){

		Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(*out_);
		muint_t n = UTX->rows();
		muint_t d = UTX->cols();
		MatrixXd XSdi(n,d);
		MatrixXd XSX(n,d);
		MatrixXd XSY(UTY->rows(), UTY->cols());
		VectorXd Sdi(n);
		Sdi = S->array() + delta;
		XSdi = UTX->array().colwise() / Sdi.array();
		XSX = XSdi.transpose() * (*UTX);
		XSY = XSdi.transpose() * (*UTY);
		out = XSX.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(XSY);
}

template <typename Derived1,typename Derived2>
inline mfloat_t estimate_bias(const Eigen::MatrixBase<Derived1>* UTy,
		const Eigen::MatrixBase<Derived2>* UT,
		const Eigen::MatrixBase<Derived1>* S,
		mfloat_t delta){

	muint_t n = UTy->rows();
	MatrixXd UT1(n,1);
	MatrixXd tmp(n,1);
	MatrixXd beta(1,1);
	tmp.setOnes();
	UT1 = (*UT) * tmp;
	ml_beta(&beta, &UT1, UTy, S, delta);
	return beta(0,0);
}

template <typename Derived1,typename Derived2, typename Derived3, typename Derived4>
inline void ml_sigma(const Eigen::MatrixBase<Derived1>& out_,
		const Eigen::MatrixBase<Derived2>& UTX,
		const Eigen::MatrixBase<Derived3>& UTY,
		const Eigen::MatrixBase<Derived4>& Sdi,
		mfloat_t delta){

	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(out_);
	MatrixXd sigma;
	MatrixXd beta;
	muint_t n = UTX.rows();
	muint_t d = UTX.cols();
	MatrixXd XSdi(n,d);
	MatrixXd XSX(n,d);
	MatrixXd XSY(UTY.rows(), UTY.cols());
	XSdi = UTX.array().colwise() * Sdi.array();
	//XSX = XSdi.transpose() * UTX;
	XSX = (UTX.transpose()*XSdi).transpose();
	//std::cout << "XSX \n" << XSX << "\n";
	XSY = XSdi.transpose() * UTY;
	//beta = XSX.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(XSY);
	beta = XSX.ldlt().solve(XSY);
	sigma = UTX*beta - UTY;
	sigma.array() *= sigma.array();
	//std::cout << "sigma_g:\n" << sigma << "\n\n";
	sigma.array().colwise() *= Sdi.array();
	//std::cout << "sigma_g:\n" << sigma << "\n\n";
	out = sigma.array().colwise().sum();
	//std::cout << "sigma_g:\n" << out << "\n\n";
	out.array() /= UTY.rows();
}

inline mfloat_t unif_rand(){
	return (mfloat_t)(rand() / double(RAND_MAX));
}

inline VectorXi sample_dimensions(muint_t max_features, VectorXi* mind){
	VectorXi rmind(max_features);
	muint_t last = mind->rows()-1;
	muint_t j;
	muint_t temp;
	for (muint_t i=0; i<max_features; ++i){
		j =  (muint_t)(unif_rand()*(last+1));
		rmind[i] = (*mind)(j);
		temp = (*mind)(last);
		(*mind)(last) = (*mind)(j);
		(*mind)(j) = temp;
		last--;
	}
	return rmind;
}

inline VectorXi get_ancestors(
		muint_t node_ind,
		muint_t node,
		VectorXi* parents){

	VectorXi ancestors(floor(log2(node+1))+1);
	ancestors(0) = node_ind;
	for (muint_t i=1; i<(muint_t)ancestors.rows(); ++i){
		node_ind = (*parents)(node_ind);
		ancestors[i] = node_ind;
	}
	return ancestors;
}
template <typename Derived1>
inline MatrixXd get_scalars(const Eigen::MatrixBase<Derived1>& XSdi, const Eigen::MatrixBase<Derived1>& UTX){
	MatrixXd XXT(XSdi.cols(),1);
		for(muint_t i=0; i<(muint_t)XXT.rows(); ++i){
			XXT(i,0) = XSdi.col(i).transpose() * UTX.col(i);
		}
	return XXT;
}

inline MatrixXd get_covariates(
		muint_t node_ind,
		mint_t node,
		VectorXi* parents,
		VectorXi* subsamples,
		VectorXi* start_index,
		VectorXi* end_index){

	VectorXi node_indexes = get_ancestors(node_ind, node, parents);
	MatrixXd Covariates(subsamples->rows(), node_indexes.rows());
	Covariates.setZero();
	for (muint_t i=0; i<(muint_t)node_indexes.rows(); ++i){
		muint_t curr_ind = node_indexes(i);
		muint_t start = (*start_index)(curr_ind);
		muint_t end = (*end_index)(curr_ind);
		for (muint_t j=start; j<end; ++j){
			Covariates(j,i) = 1.0;
		}
	}
	return Covariates;
}

template <typename Derived1,typename Derived2>
inline VectorXi relevant_dimensions(
		const Eigen::MatrixBase<Derived1>* X,
		const Eigen::MatrixBase<Derived2>* rmind){

	VectorXi rmind_out = VectorXi(rmind->rows());
	muint_t size = 0;
	for (muint_t i=0; i < (muint_t)rmind->rows(); ++i){
		mfloat_t col_sum = (*X).col((*rmind)(i)).sum();
		//std::cout << "ith index: \n" << (*rmind)(i)<< "\n\n";
		//std::cout << "values of test for similar predictors: \n" << (col_sum == X->rows() || col_sum == 0) << "\n\n";
		if ((col_sum != X->rows()) && (col_sum != 0.0)){
			//std::cout << "current  index: \n" << size<< "\n\n";
			rmind_out(size)= (*rmind)(i);
			size++;
		}
	}
	//rmind_out = rmind_out.head(size);// << "\n\n";
	return rmind_out.head(size);;
}

template <typename Derived1>
inline void transform_predictors(
		std::vector<muint_t>* feature_map,
		std::vector<mfloat_t>* splitting_level,
		Eigen::MatrixBase<Derived1>* X_in){
	vector<mfloat_t> all_levels(0);
	vector<muint_t> all_pred(0);
	muint_t cnt = 0;
	//std::cout << "number of columns "<< cnt << "\n";
	for (muint_t j=0; j<(muint_t)X_in->cols(); ++j){
		MatrixXd cc = X_in->col(j);
		mfloat_t* x = &(cc.col(0))(0);
		std::vector<mfloat_t> levels(x, x+cc.rows());
		std::vector<mfloat_t>::iterator it;
		std::sort(levels.begin(),levels.begin()+cc.rows());
		it = std::unique(levels.begin(), levels.begin()+cc.rows());
		levels.resize(std::distance(levels.begin(),it));
		cnt += levels.size()-1;
		for(it =levels.begin(); it < levels.end()-1; it++){
			mfloat_t median_level = (*(it+1) - *it) /2.0 + *it;
			all_levels.push_back(median_level);
			all_pred.push_back(j);
			//std::cout<< median_level<< "\n"; // prints d.
		}
	}
	//std::cout << "done finding cutoffs \n";
	//std::cout << "number of columns "<< cnt << "\n";
	MatrixXd X_out(X_in->rows(), cnt);
	std::vector<muint_t>::iterator it_pred; //additional pointer for knowing which predictor was the origin
	it_pred = all_pred.begin();
	muint_t ind_j = 0;
	for(std::vector<mfloat_t>::iterator it_lev=all_levels.begin(); it_lev != all_levels.end(); it_lev++, it_pred++, ind_j++){
				//std::cout<< *it_pred<< "\n"; // prints d.
				//MatrixXd cct = X_in->col(*it_pred);
				for(muint_t i=0; i<(muint_t)X_in->rows(); ++i){ //fill matrix
					if ((*X_in)(i,*it_pred) < *it_lev){
						X_out(i,ind_j) = 0.0;
					}
					else{
						X_out(i,ind_j) = 1.0;
					}
				}
				//std::cout << cct.array() < 1.0 << "\n";
				//std::cout << "check \n";
	}
	*X_in = X_out;
	*feature_map = all_pred;
	*splitting_level = all_levels;
}


template <typename Derived1,typename Derived2, typename Derived3, typename Derived4>
inline void best_split_full_model(
		mint_t* m_best,
		mfloat_t* s_best,
		mfloat_t* left_mean,
		mfloat_t* right_mean,
		mfloat_t* ll_score,
		const Eigen::MatrixBase<Derived4>* X,
		const Eigen::MatrixBase<Derived2>* UTy,
		const Eigen::MatrixBase<Derived1>* C,
		const Eigen::MatrixBase<Derived2>* S,
		const Eigen::MatrixBase<Derived1>* U,
		const Eigen::MatrixBase<Derived3>* noderange,
		mfloat_t delta){
		*m_best = -1;
		muint_t best_sub_index;
		muint_t better_split_found = 0;
		MatrixXd U_sub(noderange->rows(), UTy->rows());
		MatrixXd X_sub(noderange->rows(), X->cols());
		VectorXd Sdi(S->rows());
		Sdi.setOnes();
		Sdi.array() /= S->array() + delta; //can be done in python

		eigenIndex(U_sub, *U, *noderange);
		//UT_sub.transposeInPlace();
		eigenIndex(X_sub, *X, *noderange); //300
		std::vector<muint_t> features;
		std::vector<mfloat_t> split_level;
		transform_predictors(&features, &split_level, &X_sub);
		if (X_sub.size() == 0){
			//std:cout << "nothing to do \n";
			return; //nothing to do
		}

		MatrixXd UTC = (*U).transpose() * (*C); // 166 could  be further optimized here
		MatrixXd UTX = U_sub.transpose() * X_sub;
		MatrixXd XSdi = UTX.array().colwise() * Sdi.array();
		MatrixXd XXT = get_scalars(XSdi, UTX);
		MatrixXd CSdi = UTC.array().colwise() * Sdi.array();
		MatrixXd CSY = CSdi.transpose() * (*UTy);
		MatrixXd XSY = XSdi.transpose() * (*UTy);
		MatrixXd XSC = XSdi.transpose() * UTC;
		MatrixXd CSC = CSdi.transpose() * UTC;
		MatrixXd score_0(1,1);
		MatrixXd score(1,1); //18
		ml_sigma(score_0, UTC, (*UTy), Sdi, delta);
		mfloat_t best_score = (mfloat_t)score_0(0,0);
		MatrixXdRM XCSX(CSC.rows()+1, CSC.cols()+1);
		XCSX.block(1,1,CSC.rows(),CSC.cols()) = CSC;
		MatrixXd XCSY(CSY.rows()+1,1);
		MatrixXd beta(CSY.rows()+1,1);
		MatrixXd sigma(beta.rows(),1);
		MatrixXd UTXC(UTC.rows(),UTC.cols()+1);
		UTXC << UTX.col(0), UTC;

		XCSY.block(1,0,CSY.rows(),1) = CSY;
		//std::cout << "XXT \n" << XXT<< "\n"; //18
		for(muint_t j=0; j < (muint_t)X_sub.cols(); ++j){
			UTXC.col(0) = UTX.col(j);
			//use precomputed matrixes
			XCSX.block(0,1,CSC.rows()+1,CSC.cols()).row(0) = XSC.row(j);
			XCSX.block(1,0,CSC.rows(),CSC.cols()+1).col(0) = XSC.row(j);
			XCSX(0,0) = XXT(j,0);
			XCSY(0,0) = XSY(j);
			beta = XCSX.ldlt().solve(XCSY);
			sigma = UTXC*beta - (*UTy);
			sigma.array() *= sigma.array();
			sigma.array().colwise() *= Sdi.array();
			score = sigma.array().colwise().sum();
			score(0,0) /= UTy->rows();
			if (score(0,0)<best_score){
				better_split_found = 1;
				best_score = (mfloat_t)score(0,0);
				best_sub_index = j;
			}
		} //15
		if (better_split_found){
			UTXC.col(0) = UTX.col(best_sub_index);
			*m_best = *(features.begin() + best_sub_index);
			*s_best = *(split_level.begin() + best_sub_index);
			MatrixXd beta(1,2);
			ml_beta(&beta, &UTXC, UTy, S, delta);
			VectorXd CX = VectorXd::Zero(UTy->rows(),1);
			mint_t ind_1;
			mint_t ind_0;
			for (mint_t j=0; j<(mint_t)noderange->rows(); ++j){
				CX((*noderange)(j)) = X_sub(j,(best_sub_index));
				if (X_sub(j,(best_sub_index)) == 0.0){
					ind_0 = (*noderange)(j);
				}
				else{
					ind_1 = (*noderange)(j);
				}
			}
			MatrixXd C_new(C->rows(), C->cols()+1);
			MatrixXd C_old = (*C);
			C_new << CX, C_old;
			MatrixXd mean = C_new * beta;
			*left_mean = mean(ind_0,0);
			*right_mean = mean(ind_1,0);
			//compute likelihood scores

			*ll_score = log((mfloat_t)score_0(0,0))-log(best_score); //Difference in LL (equivalent to LOD score)
		}
}

/************************************
*** methods needed for prediction ***
*************************************/

template <typename Derived1, typename Derived2, typename Derived3>
inline void predict_rec(mfloat_t* response,
		mint_t node_ind,
		const Eigen::MatrixBase<Derived1>* tree_nodes,
		const Eigen::MatrixBase<Derived1>* left_children,
		const Eigen::MatrixBase<Derived1>* right_children,
		const Eigen::MatrixBase<Derived1>* best_predictor,
		const Eigen::MatrixBase<Derived2>* mean,
		const Eigen::MatrixBase<Derived2>* splitting_value,
		const Eigen::MatrixBase<Derived3>* X,
		mfloat_t depth)
{
	mfloat_t curr_depth =  (mfloat_t)floor(log2((*tree_nodes)(node_ind,0)+1));
	/*
	std::cout << "node number is" << (*tree_nodes)(node_ind,0);
	std::cout << "node index is " << node_ind << "\n";
	std::cout << "current depth is " << curr_depth << "\n";
	std::cout << "maximal depth is " << depth << "\n";*/
	if((*left_children)(node_ind,0) == 0 or curr_depth == depth){ //Check for leaf and depth
		*response = (*mean)(node_ind,0);
		/*
		std::cout << "current depth is " << curr_depth << "\n";
		std::cout << "node_index is " << node_ind << "\n";
		std::cout << "response by node is " << (*mean)(node_ind,0) << "\n";*/
		return;
	}
	else{
		mint_t new_node_ind;
		/*
		std::cout << "selected dimension: " << (*best_predictor)(node_ind, 0) << "\n";
		std::cout << "X " << (*X) << "\n";
		std::cout << "selected predictor value: " << (*X)(0,(*best_predictor)(node_ind, 0)) << "\n";
		std::cout << "selected splitting value: " << (*splitting_value)(node_ind, 0) << "\n";*/
		if ((*X)((*best_predictor)(node_ind, 0),0) < (*splitting_value)(node_ind, 0)){
			new_node_ind = (*left_children)(node_ind,0);
			//std::cout << "left \n";
		}
		else{
			new_node_ind = (*right_children)(node_ind,0);
			//std::cout << "right \n";
		}
		return predict_rec(response, new_node_ind, tree_nodes, left_children, right_children, best_predictor, mean, splitting_value, X, depth);
	}
}

template <typename Derived1, typename Derived2>
inline void predict(const Eigen::MatrixBase<Derived2>* response,
		const Eigen::MatrixBase<Derived1>* tree_nodes,
		const Eigen::MatrixBase<Derived1>* left_children,
		const Eigen::MatrixBase<Derived1>* right_children,
		const Eigen::MatrixBase<Derived1>* best_predictor,
		const Eigen::MatrixBase<Derived2>* mean,
		const Eigen::MatrixBase<Derived2>* splitting_value,
		const Eigen::MatrixBase<Derived2>* X,
		mfloat_t depth){

	Eigen::MatrixBase<Derived2>& response_ = const_cast< Eigen::MatrixBase<Derived2>& >(*response);

	for (mint_t i=0; i<(mint_t)X->rows(); ++i){
		mfloat_t res = 0.0;
		MatrixXd x = X->row(i);
		//std::cout << "res " << res << "\n";
		predict_rec(&res, 0, tree_nodes, left_children, right_children, best_predictor, mean, splitting_value, &x, depth);
		response_(i,0) = res;
	}
}

template <typename Derived1,typename Derived2>
inline void test(const Eigen::MatrixBase<Derived1>* out_, const Eigen::MatrixBase<Derived2>* Xr, const Eigen::MatrixBase<Derived2>* yR)
{
	Eigen::MatrixBase<Derived1>& out = const_cast< Eigen::MatrixBase<Derived1>& >(*out_);

	out = 2.0*(*Xr);

	std::cout << Xr->rows() << "," << Xr->cols() << "\n";
	std::cout << yR->rows() << "," << yR->cols() << "\n";
	std::cout << out.rows() << "," << out.cols() << "\n";

	std::cout << *Xr << "\n";
	std::cout << *yR << "\n";
}
#endif /* SPLITTINGCORE_H_ */
