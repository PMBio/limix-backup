#pragma once
#if !defined( CPlinkAlternatePhenotypFile_h )
#define CPlinkAlternatePhenotypFile_h
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
 * CPlinkAlternatePhenotypeFile - {PLINK Alternate Phenotype File Class}
 *
 *         File Name:   CPlinkAlternatePhenotypeFile.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file declare the CPlinkAlternatePhenotypeFile
 *                      class for FastLmmC
 *
 *    Change History:   
 *
 */

#include "Cplink.h"
/*
 * 'Publish' our defines
 */

/*
 * 'Publish' our class declarations / function prototypes
 */
namespace plink {
	struct AlternatePhenotypeParameters
	{
		int    fUseAlternatePhenotype;            // Using PLINK alternative phenotype file?
		size_t selectedPhenotypeIndex;
		std::string selectedPhenotypeId;
		std::string alternatePhenotypeFilename;

		AlternatePhenotypeParameters() { fUseAlternatePhenotype = 0; selectedPhenotypeIndex = 0; }
	};

	struct AlternatePhenotypeRecord
	{
		std::string idFamily;
		std::string idIndividual;
		std::vector<limix::mfloat_t> phenotypes;
	};

	class CPlinkAlternatePhenotypeFile
	{
	public:
		CPlinkAlternatePhenotypeFile(const std::string& alternatePhenotypeFilename);
		CPlinkAlternatePhenotypeFile();
		~CPlinkAlternatePhenotypeFile();

		limix::mfloat_t    PhenotypeValue(const std::string& idFamily, const std::string& idIndividual, const std::string& phenotypeLabel);
		limix::mfloat_t    PhenotypeValue(const std::string& idFamily, const std::string& idIndividual, size_t phenotypeIndex);
		limix::mfloat_t    PhenotypeValue(const std::string& individualKey, size_t phenotypeIndex);
		limix::mfloat_t    PhenotypeValue(size_t individualIndex, size_t phenotypeIndex);
		limix::mfloat_t    PhenotypeValue(const std::string& individualKey);

		size_t  IndexFromPhenotypeLabel(const std::string& phenotypeLabel);
		size_t  IndexFromIdFamilyAndIdIndividual(const std::string& idFamily, const std::string& idIndividual);
		size_t  IndexFromIndividualKey(const std::string& key);
		void    KeyFromIndividualIndex(size_t idx, std::string& keyOut);

		size_t  CountOfPhenotypes()  { return(cPhenotypes); }
		size_t  CountOfIndividuals() { return(cIndividuals); }

		void    Read();
		void    Read(const std::string& alternatePhenotypeFilename);
		void    SetSelectedPhenotype(std::string& idPhenotype);
		void    SetSelectedPhenotype(size_t idx);
		size_t  SelectedPhenotypeIndex() { return(selectedPhenotype); }
		std::string& SelectedPhenotypeName();
		std::string& PhenotypeName(size_t idx);

		std::vector< std::string > KeysForIndividualsWithValidPhenotypes(size_t idx);
		std::vector< std::string > KeysForIndividualsWithValidPhenotypes(const std::string& idPhenotype);

	private:
		std::string  filename;
		std::map< std::string, size_t > phenotypeIndividualIdIndex;
		std::map< std::string, size_t > phenotypeLabelIndex;

		std::vector< std::string > phenotypeNames;
		std::vector< AlternatePhenotypeRecord > records;
		size_t  cIndividuals;         // count of individuals in alternate phenotype file
		size_t  cPhenotypes;          // count of Phenotypes per individual
		size_t  selectedPhenotype;    // the 'selected' phenotype is the one we are studying this run

		void Init() { cIndividuals = cPhenotypes = selectedPhenotype = 0; }
	};
} //end :plink
#endif
