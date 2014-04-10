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

#include "CCovariatesData.h"

namespace plink {

	CCovariatesData::CCovariatesData()
	{
		n_individuals = 0;
		n_covariates = 0;
	}

	CCovariatesData::~CCovariatesData()
	{
	}

	/*
	 *  Read a PLink formated Covariate file into this object
	 */
	void CCovariatesData::ReadCovariatesFile(std::string& covariatesFile)
	{
		filename = covariatesFile;
		n_individuals = 0;
		CToken tok;
		std::vector<limix::mfloat_t> covariatesCurrent;     // keep constructor out of loop

		CPlinkLexer lex(covariatesFile);
		lex.NextToken(tok);

		while (tok.type != tokEOF)        // for each line until EOF
		{
			if (tok.type == tokEOL)        // skip blank line...or comment line or ???
			{
				lex.NextToken(tok);
				continue;
			}

			std::string idFamily;
			std::string idIndividual;
			limix::mfloat_t covarCurrent;
			int covarCurrentValueType;

			lex.ExpectId(tok, idFamily, "FamilyID");
			famIDs.push_back(idFamily);
			lex.ExpectId(tok, idIndividual, "IndividualID");
			indIDs.push_back(idIndividual);

			covariatesCurrent.clear();       // start this iteration w/ a 'clean' std::vector

			while ((tok.type != tokEOL) && (tok.type != tokEOF))
			{
				lex.ExpectPhenotype(tok, covarCurrent, covarCurrentValueType);
				covariatesCurrent.push_back(covarCurrent);
			}

			if ((n_individuals > 0) && (covariatesCurrent.size() != n_covariates))
			{
				Fatal("\nExpected  %i covariates, found %i", n_covariates, covariatesCurrent.size());
			}
			else if (n_individuals == 0)
			{
				n_covariates = covariatesCurrent.size();
			}

			covariates.push_back(covariatesCurrent);

			std::string key = KeyFromIdFamilyAndIdIndividual(idFamily, idIndividual);  // create a unique key for this individual
			this->keyIndividualToCovariatesDataIndividualIndex[key] = n_individuals;
			++n_individuals;
		}
	}

	//looks for constants among the covariates and removes them from the analysis.
	void CCovariatesData::MarkConstantCovariates()
	{
		for (size_t cov = 0; cov < n_covariates; ++cov)
		{
			bool onlyNaN = true;    //so far we have observed only NaN for the feature
			bool constant = true;   //In the beginning we expect the feature to be constant
			limix::mfloat_t val;               //this holds the first value found.

			for (size_t ind = 0; ind < n_individuals; ++ind)
			{
				if (onlyNaN)
				{
					//This is the case, if all individuals up to here are NaN
					if (covariates[ind][cov] == covariates[ind][cov])
					{
						//not NaN, we have the first 'observed value'
						val = covariates[ind][cov];
						onlyNaN = false;    //At least one observed value exists.
					}
				}
				else
				{
					//There has already been an observed value.
					if ((covariates[ind][cov] == covariates[ind][cov])
						&& (val != covariates[ind][cov]))
					{
						//not NaN and the covariate is not constant.
						//  It will be used in the further analysis.
						constant = false;
						break;
					}
				}
			}//end for loop over individuals

			if (constant)
			{
				//TODO: somehow remove the covariate
				if (onlyNaN)
				{
					Warn("Covariate number %i consists only of NaN values", cov);
				}
				else
				{
					Warn("Covariate number %i is constant", cov);
				}
			}
		}
	}


	void CCovariatesData::CreateColumnMajorArrayFromVectorsAndMapping(std::vector<std::string>& individualsLabelOrder, limix::mfloat_t *covariatesArray, size_t n_covariates_)
	{
		//   if ( individualsLabelOrder.size() != n_individuals )
		//      {
		//      Warn( "Covariates File contains %i individuals, Only %i observed phenotypes are specified.", n_individuals, individualsLabelOrder.size() );
		//      }
		if (n_covariates_ != n_covariates)
		{
			Fatal("Covariates file contains %i covariates, size of array is specified as %i covariates", n_covariates_, n_covariates);
		}

		size_t cIndividualsOut = individualsLabelOrder.size();
		std::string keyIndividual;
		for (size_t iIndividual = 0; iIndividual < cIndividualsOut; ++iIndividual)
		{
			keyIndividual = individualsLabelOrder[iIndividual];
			if (!FKeyIndividualInDatFile(keyIndividual))
			{
				Fatal("Cannot create column major array from Vector and Mapping.  individualIDs of covariates do not contain mapping key [%s]", keyIndividual.c_str());
			}

			size_t i = keyIndividualToCovariatesDataIndividualIndex[keyIndividual];
			for (size_t covar = 0; covar < this->n_covariates; ++covar)
			{
				covariatesArray[iIndividual + (cIndividualsOut * covar)] = covariates[i][covar];
			}
		}
	}

	size_t CCovariatesData::IdxFromKeyIndividual(const std::string& keyIndividual)
	{
		if (keyIndividualToCovariatesDataIndividualIndex.count(keyIndividual) == 0)
		{
			Fatal("Unable to find individual [%s] in Covariates data.", keyIndividual.c_str());
		}
		size_t idx = keyIndividualToCovariatesDataIndividualIndex[keyIndividual];
		return(idx);
	}

	size_t CCovariatesData::IdxFromIdFamilyAndIdIndividual(const std::string& idFamily, const std::string& idIndividual)
	{
		std::string key = KeyFromIdFamilyAndIdIndividual(idFamily, idIndividual);
		size_t idx = IdxFromKeyIndividual(key);
		return(idx);
	}

	bool CCovariatesData::FKeyIndividualInDatFile(const std::string& keyIndividual)
	{
		return(keyIndividualToCovariatesDataIndividualIndex.count(keyIndividual) != 0);
	}

}// end :plink