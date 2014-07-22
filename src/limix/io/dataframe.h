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

#ifndef DATAFRAME_H_
#define DATAFRAME_H_

#include "limix/types.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>


namespace limix {


template <int rows,int cols,int options>
/*!
 * Wrapper to handle Eigen arrays of flexible type.
 *
 * This is particularly used for row and header handling, which have unknown data types,
 * See \ref CHeaderMap for details.
 */
class FlexEigenMatrix
{
public:
	//<! Currently, this class supports either INT, DOUBLE or STRING types
	enum FlexType { INT, FLOAT, STRING, NONE };
	//definitions of the corresponding Eigen types for convenience
	typedef Eigen::Matrix<mint_t,rows,cols,options> IntMatrix;
	typedef sptr<IntMatrix> PIntMatrix;
	typedef Eigen::Matrix<mfloat_t,rows,cols,options> FloatMatrix;
	typedef sptr<FloatMatrix> PFloatMatrix;
	typedef Eigen::Matrix<std::string,rows,cols,options> StringMatrix;
	typedef sptr<StringMatrix> PStringMatrix;

protected:
	/*!
	 * typeCheck operates in two modes. By defatul we assume we access the dataset (set=False)
	 * If data access (setting) is requested, set =true.
	 */
	void typeCheck(FlexType type,bool set=false) 
	{
		if(set)
		{
			if((this->type!=type) && (this->type!=NONE))
				throw CLimixException("Type check, flexible EigenArrays need to have a predetermined type which is constant across their lifetime");
			this->type = type;
		}
		else
		{
			if((this->type!=type))
				throw CLimixException("Type check failed, attempted to convert a flexible array into an incompatible type");
		}
	}
	PVoid array;    //<! The underlying pointer to the unknown Eigen Type
	FlexType type; 	//<! The type of the underlying array
public:

	FlexEigenMatrix()
	{
		this->type = NONE;
	};

	FlexEigenMatrix(FlexType type)
	{
		this->type = type;
		switch(type)
		{
			case(INT):
				this->array = PIntMatrix(new IntMatrix());
				break;
			case(FLOAT):
				this->array = PFloatMatrix(new FloatMatrix());
				break;
			case(STRING):
				this->array = PStringMatrix(new StringMatrix());
				//this->array = PStringMatrix(new StringMatrix()));
				break;
		}
	};

	FlexEigenMatrix(PIntMatrix array)
	{
		this->array = array;
		this->type = INT;
	};

	FlexEigenMatrix(const IntMatrix& array)
	{
		this->array = PIntMatrix(new IntMatrix(array));
		this->type = INT;
	}

	FlexEigenMatrix(PFloatMatrix array)
	{
		this->array = array;
		this->type = FLOAT;
	};

	FlexEigenMatrix(const FloatMatrix& array)
	{
		this->array = PFloatMatrix(new FloatMatrix(array));
		this->type = FLOAT;
	}

	FlexEigenMatrix(PStringMatrix array)
	{
		this->array = array;
		this->type = STRING;
	};

	FlexEigenMatrix(const StringMatrix& array)
	{
		this->array = PStringMatrix(new StringMatrix(array));
		this->type = STRING;
	}
	//wrapper for key eigen functions
    void conservativeResize(muint_t size)
    {
    	if(this->type==STRING)
    		static_pointer_cast<StringMatrix>(this->array)->conservativeResize(size);
    }

	//getter and setter
	FlexType getType()
	{ return this->type;};
	bool isType(FlexType type)
	{ return this->type==type;}

	void AgetM(IntMatrix* out) 
	{
		typeCheck(INT);
		(*out) =(*static_pointer_cast<IntMatrix>(this->array));
	}
	void AgetM(FloatMatrix* out) 
	{
		typeCheck(FLOAT);
		(*out) =(*static_pointer_cast<FloatMatrix>(this->array));
	}
	void AgetM(StringMatrix* out) 
	{
		typeCheck(STRING);
		(*out) =(*static_pointer_cast<StringMatrix>(this->array));
	}

	operator PIntMatrix() 
	{
		typeCheck(INT);
		return static_pointer_cast<IntMatrix>(this->array);
	}

	operator IntMatrix() 
	{
		typeCheck(INT);
		return *static_pointer_cast<IntMatrix>(this->array);
	}


	operator FloatMatrix() 
	{
		typeCheck(FLOAT);
		return *static_pointer_cast<FloatMatrix>(this->array);
	}


	operator PFloatMatrix() 
	{
		typeCheck(FLOAT);
		return static_pointer_cast<FloatMatrix>(this->array);
	}



	operator PStringMatrix() 
	{
		typeCheck(STRING);
		return static_pointer_cast<StringMatrix>(this->array);
	}

	operator StringMatrix() 
	{
		typeCheck(STRING);
		return *static_pointer_cast<StringMatrix>(this->array);
	}

	void operator= (PIntMatrix array) 
	{
		setM(array);
	}
	void operator= (const IntMatrix& array) 
	{
		setM(array);
	}

	void operator= (PFloatMatrix array) 
	{
		setM(array);
	}
	void operator= (const FloatMatrix& array) 
	{
		setM(array);
	}

	void operator= (PStringMatrix array) 
	{
		setM(array);
	}
	void operator= (const StringMatrix& array) 
	{
		setM(array);
	}

	void setM(PIntMatrix array) 
	{
		typeCheck(INT,true);
		this->array =array;
	}
	void setM(PFloatMatrix array) 
	{
		typeCheck(FLOAT,true);
		this->array =array;
	}
	void setM(PStringMatrix array) 
	{
		typeCheck(STRING,true);
		this->array =array;
	}
	void setM(const FloatMatrix& array) 
	{
		typeCheck(FLOAT,true);
		this->array = new PFloatMatrix(new FloatMatrix(array));
	}
	void setM(const IntMatrix& array) 
	{
		typeCheck(INT,true);
		this->array = new PIntMatrix(new IntMatrix(array));
	}
	void setM(const StringMatrix& array) 
	{
		typeCheck(STRING,true);
		this->array = new PStringMatrix(new StringMatrix(array));
	}

	//modification operators

};
typedef FlexEigenMatrix<Eigen::Dynamic, 1, Eigen::ColMajor> CFlexVector;
typedef FlexEigenMatrix<Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> CFlexMatrix;
typedef sptr<CFlexVector> PFlexVector;
typedef sptr<CFlexMatrix> PFlexMatrix;


/*!
 *  Flexible object containing header objects
 *
 */
class CHeaderMap;
typedef sptr<CHeaderMap> PHeaderMap;

/*! \brief Map of keys for usage as a hader
 *
 *	CHeaderMap is based on a map of pointers to vectors of strings
 *	Each element consists of a vector with header elemnts which are used as columns or rows
 */
class CHeaderMap : public std::map<std::string,CFlexVector>
{
public:
	/*!
	 * resize the internal vectors for all elements
	 */
	void resize(muint_t n);
	/*!
	 * directly set the value of a header vectors
	 * \param name: header key
	 * \param n: index
	 * \param value: string value
	 */
	void setStr(std::string name, muint_t n, std::string value);
	/*!
	 * directly retrieve the value of a particular header element, assuming the column is a string type
	 * \param name: header key
	 * \param n: index
	 */

    //TODO: fix me	 
	//std::string get(std::string name,muint_t n);
	/*!
	 * Copy a certain number of elements form the header map
	 * \param i_start: start index
	 * \param n_elements: number of elements to copy
	 */
	PHeaderMap copy(muint_t i_start,muint_t n_elements);

	/*!
	* Copy a certain number of elements form the header map
	* \param n_elements: number of elements to copy
	*/
	PHeaderMap copy(muint_t n_elements)
	{
		return copy(0,n_elements);
	}
};



/*!
 * Abstract base class for read only data frame
 */
template <class MatrixType>
class ARDataFrame {
public:

	ARDataFrame()
	{};
	virtual ~ARDataFrame()
	{};

	//virtual void agetRowHeader(VectorXs* out) const  = 0;
	virtual PHeaderMap getRowHeader() const  = 0;

	//virtual void agetColHeader(VectorXs* out) const  = 0;
	virtual PHeaderMap getColHeader() const  = 0;

	virtual void agetMatrix(MatrixType* out) const  = 0;
	virtual sptr<MatrixType> getMatrix() const  =0;

};

/*!
 * Abstract base class for write only data frame
 */
template <class MatrixType>
class AWDataFrame {
public:
	AWDataFrame()
	{};
	virtual ~AWDataFrame()
	{};

	virtual void setRowHeader(PHeaderMap in)  = 0;
	virtual void setColHeader(PHeaderMap in)  = 0;

	virtual void setMatrix(const MatrixType& in)  = 0;
	virtual void setMatrix(sptr<MatrixType> in)  = 0;
};


/*!
 * In memory implementation of read only dataframe
 */
template <class MatrixType>
class CRMemDataFrame : public ARDataFrame<MatrixType>
{
protected:
	sptr<MatrixType> M;
	//StringVec
	PHeaderMap rowHeader,colHeader;

	virtual void resizeMatrices(muint_t num_samples=-1,muint_t num_snps=-1);

public:
	CRMemDataFrame()
	{
		M = sptr<MatrixType>(new MatrixType());
		rowHeader = PHeaderMap(new CHeaderMap());
		colHeader = PHeaderMap(new CHeaderMap());
	};
	CRMemDataFrame(const CRMemDataFrame<MatrixType>& copy) : ARDataFrame<MatrixType>()
	{
		this->M = PMatrixXd(new MatrixXd(*copy.M));
		this->rowHeader = PHeaderMap(new CHeaderMap(*copy.rowHeader));
		this->colHeader = PHeaderMap(new CHeaderMap(*copy.colHeader));
	};


	CRMemDataFrame(sptr<MatrixType> M,PHeaderMap rowHeader,PHeaderMap colHeader)
	{
		this->M = M;
		this->colHeader = colHeader;
		this->rowHeader = rowHeader;
	};

	virtual ~CRMemDataFrame()
	{};
	virtual PHeaderMap getRowHeader() const 
	{
		return rowHeader;
	}
	virtual PHeaderMap getColHeader() const 
	{
			return colHeader;
	}

	virtual void agetMatrix(MatrixType* out) const 
	{
		(*out) = *M;
	}
	virtual sptr<MatrixType> getMatrix() const 
	{
		return M;
	}
};

//implementation for in-memory handling READ ONLY
template <class MatrixType> class CRWMemDataFrame : public CRMemDataFrame<MatrixType>, public AWDataFrame<MatrixType>
{
public:
	CRWMemDataFrame()
	{};
	CRWMemDataFrame(const CRMemDataFrame<MatrixType>& copy) : CRMemDataFrame<MatrixType>(copy)
	{};

	virtual ~CRWMemDataFrame()
	{};

	virtual void setRowHeader(PHeaderMap in) 
		{
		(this->rowHeader) = in;
		}
	virtual void setColHeader(PHeaderMap in) 
		{
		(this->colHeader) = in;
		}

	virtual void setMatrix(const MatrixType& in) 
		{
		(*this->M) = in;
		}
	virtual void setMatrix(sptr<MatrixType> in) 
		{
		this->M = in;
		}

};

}
		//end: namespace limix

template<class MatrixType>
inline void limix::CRMemDataFrame<MatrixType>::resizeMatrices(
		muint_t num_rows, muint_t num_columns) {
	this->M->conservativeResize(num_rows,num_columns);
	this->rowHeader->resize(num_rows);
	this->colHeader->resize(num_columns);
	this->rowHeader->resize(num_rows);
	this->colHeader->resize(num_columns);
}

#endif /* DATAFRAME_H_ */
