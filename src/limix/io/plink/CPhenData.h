#if !defined( CPhenData_h )
#define CPhenData_h
/*
 *******************************************************************
 *
 *    Copyright (c) Microsoft. All rights reserved.
 *    This code is licensed under the Apache License, Version 2.0.
 *    THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
 *    ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
 *    IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
 *    PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
 *
 ******************************************************************
 */

/*
 * CPhenData - {Phenotype Data Class}
 *
 *         File Name:   CPhenData.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file declares the CPhenData class for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Get our includes
 */
#include "Cplink.h"
/*
 * 'Publish' our defines
 */

/*
 * 'Publish' our class declarations / function prototypes
 */
namespace plink {

	class CPhenData
	{
	public:
		std::vector<std::string> rowLabels;
		std::vector<std::string> columnLabels;
		std::vector<std::vector<limix::mfloat_t> > phenVectors;       // temporary storage of Phenotype reads

		size_t         cRows;
		size_t         cColumns;
		std::map<std::string, size_t> columnMap;      // map column to column index
		std::map<std::string, size_t> rowMap;         // map column to column index

		limix::mfloat_t         *phenArray;

		CPhenData();
		~CPhenData();
		void ReadPhenFile(std::string& phenFile);
		void WritePhenFile(std::string& phenFile);
		void CreateColumnMajorArrayFromVectors();
		void CreateColumnMajorArrayFromVectorsAndMapping(std::vector<std::string> columnLabelOrder);
	};


	/*
	 * 'Publish' the globals we define
	 */
}//end: plink
#endif   // CPhenData_h
