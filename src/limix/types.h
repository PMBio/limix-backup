// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#ifndef TYPES_H_
#define TYPES_H_

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <exception>

#define CPP11

#ifdef _WIN32
    #include <unordered_map>
    #include <memory>
	#include<limits>
//use new C++ 11 way of defining things:
#elif defined(CPP11)
	#include <memory>
#else
    #include <tr1/memory>
#endif

//define shortcut for shared pointer
#if defined(CPP11)
#define sptr std::shared_ptr
#define static_pointer_cast std::static_pointer_cast
#define dynamic_pointer_cast std::dynamic_pointer_cast
#define enable_shared_from_this std::enable_shared_from_this
#else
#define sptr std::tr1::shared_ptr
#define static_pointer_cast std::tr1::static_pointer_cast
#define dynamic_pointer_cast std::tr1::dynamic_pointer_cast
#define enable_shared_from_this std::tr1::enable_shared_from_this
#define nullptr NULL
#endif

#ifndef nullptr
#define nullptr NULL
#endif

/*
//BOOST shared pointer
#define sptr boost::shared_ptr
#define enable_shared_from_this boost::enable_shared_from_this
#define dynamic_pointer_cast boost::dynamic_pointer_cast
#define static_pointer_cast boost::static_pointer_cast
*/

namespace limix{

#ifndef PI
#define PI 3.14159265358979323846
#endif

const double L2pi = 1.8378770664093453;


//note: for swig it is important that everyhing is typed def and not merely "defined"
typedef double float64_t;
typedef float float32_t;
//get integer definitions from stdint or define manuall for MSVC


//typedef long int int64_t;
//typedef unsigned long int uint64_t;
#include <stdint.h>

/*
#ifdef _MSC_VER
 typedef __int32 int32_t;
 typedef unsigned __int32 uint32_t;
 typedef __int64 int64_t;
 typedef unsigned __int64 uint64_t;
#else
//#include <stdint.h>
typedef long int int64_t;
typedef unsigned long int uint64_t;
#endif
*/



//default types for usage in GPmix:
typedef float64_t mfloat_t;
typedef int64_t mint_t;
typedef uint64_t muint_t;


#ifdef _WIN32
    #ifndef NAN
		#define NAN std::numeric_limits<mfloat_t>::quiet_NaN()
	#endif
	#ifndef INFINITY
		#define INFINITY std::numeric_limits<mfloat_t>::infinity( )
	#endif
#endif

//simple definition of isnan
#ifndef isnan
	//#define isnan(val) val!=val
	inline bool isnan(mfloat_t val)
	{
		if (val != val)
		 return true;
		else
		 return false;
	};
#endif

#ifndef isinf
	//template<typename T>
	inline bool isinf(mfloat_t value)
	{
		return (value == std::numeric_limits<mfloat_t>::infinity());
	};
#endif


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
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXdRM;
typedef Eigen::Matrix<mfloat_t, 2, 2,Eigen::ColMajor> MatrixXd2;
typedef Eigen::Matrix<mfloat_t, 3, 3,Eigen::ColMajor> MatrixXd3;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXd;
//smart pointer objects
typedef sptr<MatrixXd> PMatrixXd;
typedef sptr<VectorXd> PVectorXd;
typedef sptr<const MatrixXd> PConstMatrixXd;
typedef sptr<const VectorXd> PConstVectorXd;
typedef sptr<void> PVoid;
typedef sptr<const void> PCVoid;


//integer
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXi;
typedef Eigen::Matrix<mint_t, 2, 2,Eigen::ColMajor> MatrixXi2;
typedef Eigen::Matrix<mint_t, 3, 3,Eigen::ColMajor> MatrixXi3;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXi;
//smart pointer objects
typedef sptr<MatrixXi> PMatrixXi;
typedef sptr<VectorXi> PVectorXi;

//boolean
typedef Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> MatrixXb;
typedef Eigen::Matrix<bool,Eigen::Dynamic,1,Eigen::ColMajor> VectorXb;

typedef sptr<MatrixXb> PMatrixXb;
typedef sptr<VectorXb> PVectorXb;


//string
typedef Eigen::Matrix<std::string, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> ArrayXs;
typedef Eigen::Matrix<std::string, Eigen::Dynamic, 1,Eigen::ColMajor> Array1DXs;
typedef sptr<ArrayXs> PArrayXs;
typedef sptr<Array1DXs> PArray1DXs;

//typedef Eigen::Array<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> ArrayXd;

//cholesky decomposition object (we use LLT as faster and more stable)
typedef Eigen::LLT<limix::MatrixXd> MatrixXdChol;
//eigen decomposition
typedef Eigen::SelfAdjointEigenSolver<MatrixXd> MatrixXdEIgenSolver;

//SCIPY matrices for python interface: these are row major
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXdscipy;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1> VectorXdscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXiscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1> VectorXiscipy;
#endif

//convenience wrappers
typedef std::vector<MatrixXd> MatrixXdVec;
typedef std::vector<std::string> stringVec;
typedef sptr<stringVec> PstringVec;


//GpMix Exception Class
class CLimixException : public std::exception
{
  public:
	CLimixException(std::string str="Unlabeled exception")
      : What(str)
    {
    }
    virtual ~CLimixException() throw ()
    {}
    virtual std::string what()
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
