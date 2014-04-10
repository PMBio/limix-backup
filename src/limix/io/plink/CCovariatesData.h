#if !defined( CCovariatesData_h )
#define CCovariatesData_h
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

#include "Cplink.h"

namespace plink {

	class CCovariatesData
	{
	public:
		std::vector<std::string> indIDs;
		std::vector<std::string> famIDs;
		std::vector<std::vector<limix::mfloat_t> > covariates;      // temporary storage of Covariates
		std::vector<bool>   markedAsConstant;       // indicator of constant covariates

		size_t         n_individuals;
		size_t         n_covariates;
		std::map< std::string, size_t > keyIndividualToCovariatesDataIndividualIndex;

		std::string         filename;

		CCovariatesData();
		~CCovariatesData();
		void ReadCovariatesFile(std::string& covariatesFile);
		void WriteCovariatesFile(std::string& covariatesFile);
		//   void CreateColumnMajorArrayFromstd::vectors();
		void CreateColumnMajorArrayFromVectorsAndMapping(std::vector<std::string>& columnLabelOrder, real *covariatesArray, size_t n_covariates);
		void MarkConstantCovariates();

		bool     FKeyIndividualInDatFile(const std::string& keyIndividual);
		size_t   IdxFromKeyIndividual(const std::string& keyIndividual);
		size_t   IdxFromIdFamilyAndIdIndividual(const std::string& idFamily, const std::string& idIndividual);

	};
}//end: plink
#endif   // CCovariatesData_h
