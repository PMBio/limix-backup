#if !defined( CKernelData_h )
#define CKernelData_h
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
 * CKernelData - {Kernel Data Class}
 *
 *         File Name:   CKernelData.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file declares the CKernelData class for FastLmmC
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

	class CKernelData
	{
	public:
		std::vector<std::string> columnLabels;
		std::vector<std::string> rowLabels;
		std::vector<std::vector<limix::mfloat_t> > kernelVectors;        // temporary storage of Kernel reads

		size_t         cColumns;
		size_t         cRows;
		std::map<std::string, size_t> columnMap;      // map column to column index
		std::map<std::string, size_t> rowMap;         // map column to column index

		limix::mfloat_t         *kernelArray;

		CKernelData();
		~CKernelData();
		void ReadKernelFile(std::string& kernelFile);
		void WriteKernelFile(std::string& kernelFile);
		void CreateColumnMajorArrayFromVectors();
		void CreateColumnMajorArrayFromVectorsAndMapping(std::vector<std::string> columnLabelOrder);
	};

	/*
	 * 'Publish' the globals we define
	 */
}//end: plink
#endif   // CKernelData_h
