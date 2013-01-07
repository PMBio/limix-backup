/*
 * types.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#ifdef _WIN32
    #include <unordered_map>
    #include <memory>
#else
//    #include <tr1/unordered_map>
    #include <tr1/memory>
#endif

//define shortcut for shared pointer
#define sptr std::tr1::shared_ptr
#define static_pointer_cast std::tr1::static_pointer_cast
#define dynamic_pointer_cast std::tr1::dynamic_pointer_cast

#define enable_shared_from_this std::tr1::enable_shared_from_this

namespace limix{

#ifndef PI
#define PI 3.14159265358979323846
#endif

const double L2pi = 1.8378770664093453;


//note: for swig it is important that everyhing is typed def and not merely "defined"
typedef double float64_t;
typedef float float32_t;
typedef long int int64_t;
typedef unsigned long int uint64_t;

//default types for usage in GPmix:
typedef float64_t mfloat_t;
typedef int64_t mint_t;
typedef uint64_t muint_t;


//inline casts of exp and log
inline mfloat_t exp (mfloat_t x)
{
		return (mfloat_t)std::exp((long double) x );
}

inline mfloat_t sqrt (mfloat_t x)
{
		return (mfloat_t)std::sqrt((long double) x );
}


inline mfloat_t log (mfloat_t x)
{
		return (mfloat_t)std::log((long double) x );
}

inline mfloat_t inverse (mfloat_t x)
{
	return 1.0/x;
}

//we exclude these int he wrapping section of SWIG
//this is somewhat ugly but makes a lot easier to wrap the Eigen arays with swig
#if (!defined(SWIG) || defined(SWIG_FILE_WITH_INIT))
//standard Matrix type to use in this project

//float
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXd;
typedef Eigen::Matrix<mfloat_t, 2, 2,Eigen::ColMajor> MatrixXd2;
typedef Eigen::Matrix<mfloat_t, 3, 3,Eigen::ColMajor> MatrixXd3;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXd;

//integer
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXi;
typedef Eigen::Matrix<mint_t, 2, 2,Eigen::ColMajor> MatrixXi2;
typedef Eigen::Matrix<mint_t, 3, 3,Eigen::ColMajor> MatrixXi3;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXi;

//string
typedef Eigen::Matrix<std::string, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXs;
typedef Eigen::Array<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> ArrayXd;
//cholesky decomposition object (we use LLT as faster and more stable)
typedef Eigen::LLT<limix::MatrixXd> MatrixXdChol;



//SCIPY matrices for python interface: these are row major
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXdscipy;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1> VectorXdscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXiscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1> VectorXiscipy;
#endif

//convenience wrappers
typedef std::vector<MatrixXd> MatrixXdVec;


//GpMix Exception Class
class CGPMixException
{
  public:
	CGPMixException(std::string str="Unlabeled exception")
      : What(str)
    {
		std::cout <<"GPMIX Exception: " << str << "\n";
    }
    std::string what()
    {
      return What;
    }

  private:
    std::string What;
};


//base class for ParameterObjects (caching)
typedef sptr<bool> Pbool;
typedef std::vector<Pbool> PboolVec;
/* CParamObject:
 * - provides basic handling for synchronization of precalculated data
 * syncParents:  boolen sync states the computation of this object depends on
 * syncChildren: children object whose calculations depend on (this)
 *
 * addSyncParent / addSyncChild: add a sync element
 * isInSync(): checks that all parent sync objects are true (i.e. does the state of hte current object need to updated?)
 * makeSync(): sets all parent sync true (i.e. set the current object to have an up2date state)
 */
class CParamObject
{
protected:
	Pbool sync;
	//paranets: objects that require sync above the hierarchy of this one
	PboolVec syncParents;
	//paranets: objects that require sync below the hierarchy of this one
	PboolVec syncChildren;
	//propagateSync: set all children sync states to false to force them update computations
public:

	CParamObject()
	{
		sync = Pbool(new bool);
		//add standard sync paranet which is the own sync variable
		addSyncParent(sync);
	}

	virtual void addSyncParent(Pbool l)
	{
		//set false
		*l = false;
		syncParents.push_back(l);
	}
	virtual void addSyncChild(Pbool l)
	{
		*l = false;
		syncChildren.push_back(l);
	}
	virtual void delSyncParent(Pbool l)
	{
		//TODO: implement me
	}
	virtual void delSyncChild(Pbool l)
	{
		//TODO: implement me
	}




	void propagateSync(bool state=false)
	{
		for(PboolVec::iterator iter = syncChildren.begin(); iter!=syncChildren.end();iter++)
			{
				(*iter[0]) = state;
			}
	}

	virtual bool isInSync()
	{
		for(PboolVec::iterator iter = syncParents.begin(); iter!=syncParents.end();iter++)
			{
				if (!(*iter[0]))
					return false;
			}
		return true;
	}

	virtual void setSync(bool state=true)
	{
		for(PboolVec::iterator iter = syncParents.begin(); iter!=syncParents.end();iter++)
		{
			*iter[0] = state;
		}
	}

	
};

}//end ::namespace limix


#endif /* TYPES_H_ */
