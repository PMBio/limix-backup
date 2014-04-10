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
 * CPlinkFile - {PLINK File Access Class}
 *
 *         File Name:   CPlinkFile.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements the CPlinkFile class for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "CPlinkAlternatePhenotypeFile.h"

namespace plink {
	CPlinkAlternatePhenotypeFile::CPlinkAlternatePhenotypeFile()
	{
		Init();
	}

	CPlinkAlternatePhenotypeFile::CPlinkAlternatePhenotypeFile(const std::string& alternatePhenotypeFilename)
	{
		Init();
		filename = alternatePhenotypeFilename;
	}

	CPlinkAlternatePhenotypeFile::~CPlinkAlternatePhenotypeFile()
	{
	}

	size_t CPlinkAlternatePhenotypeFile::IndexFromIndividualKey(const std::string& key)
	{
		size_t cMatch = phenotypeIndividualIdIndex.count(key);
		if (cMatch == 0)
		{
			// No match.  return size_t of NoMatch if no match
			return(NoMatch);
		}
		size_t individualIndex = phenotypeIndividualIdIndex[key];
		return(individualIndex);
	}

	size_t CPlinkAlternatePhenotypeFile::IndexFromIdFamilyAndIdIndividual(const std::string& idFamily, const std::string& idIndividual)
	{
		std::string key = KeyFromIdFamilyAndIdIndividual(idFamily, idIndividual);
		return(IndexFromIndividualKey(key));
	}

	void CPlinkAlternatePhenotypeFile::KeyFromIndividualIndex(size_t idx, std::string& keyOut)
	{
		/*
		 *  Return the std::string key associated with record[idx]
		 */
		if (idx >= records.size())
		{
			Fatal("Trying to access individualIndex out of range.  Max is %d, index is %d", records.size(), idx);
		}
		keyOut = KeyFromIdFamilyAndIdIndividual(records[idx].idFamily, records[idx].idIndividual);
	}

	limix::mfloat_t CPlinkAlternatePhenotypeFile::PhenotypeValue(const std::string& idFamily, const std::string& idIndividual, const std::string& phenotypeLabel)
	{
		size_t phenotypeIndex = IndexFromPhenotypeLabel(phenotypeLabel);
		return(PhenotypeValue(idFamily, idIndividual, phenotypeIndex));
	}

	limix::mfloat_t CPlinkAlternatePhenotypeFile::PhenotypeValue(const std::string& idFamily, const std::string& idIndividual, size_t phenotypeIndex)
	{
		size_t individualIndex = IndexFromIdFamilyAndIdIndividual(idFamily, idIndividual);
		if (individualIndex == NoMatch)
		{
			// return NaN if no match
			return(std::numeric_limits<limix::mfloat_t>::quiet_NaN());
		}

		return(PhenotypeValue(individualIndex, phenotypeIndex));
	}

	limix::mfloat_t CPlinkAlternatePhenotypeFile::PhenotypeValue(const std::string& individualKey, size_t phenotypeIndex)
	{
		size_t individualIndex = IndexFromIndividualKey(individualKey);
		if (individualIndex == NoMatch)
		{
			// return NaN if no match
			return(std::numeric_limits<limix::mfloat_t>::quiet_NaN());
		}

		return(PhenotypeValue(individualIndex, phenotypeIndex));
	}

	limix::mfloat_t CPlinkAlternatePhenotypeFile::PhenotypeValue(const std::string& individualKey)
	{
		return(PhenotypeValue(individualKey, selectedPhenotype));
	}

	limix::mfloat_t CPlinkAlternatePhenotypeFile::PhenotypeValue(size_t individualIndex, size_t phenotypeIndex)
	{
		if (individualIndex >= records.size())
		{
			Fatal("Trying to access individualIndex out of range.  Max is %d, index is %d", records.size(), individualIndex);
		}
		if (phenotypeIndex >= records[individualIndex].phenotypes.size())
		{
			Fatal("Trying to access phenotypeIndex out of range.  Max is %d, index is %d", records[individualIndex].phenotypes.size(), phenotypeIndex);
		}
		return(records[individualIndex].phenotypes[phenotypeIndex]);
	}

	void CPlinkAlternatePhenotypeFile::Read(const std::string& alternatePhenotypeFilename)
	{
		filename = alternatePhenotypeFilename;
		Read();
	}

	void CPlinkAlternatePhenotypeFile::Read()
	{
		Verbose("    Processing Alternate Phenotypes: [%s]", filename.c_str());
		CPlinkLexer lex(filename);
		CToken tok;
		AlternatePhenotypeRecord apr;

		cPhenotypes = 0;
		cIndividuals = 0;

		lex.NextToken(tok);
		while (tok.type != tokEOF)
		{
			if (tok.type == tokEOL)
			{
				lex.NextToken(tok);
				continue;
			}

			lex.ExpectId(tok, apr.idFamily, "FamilyID");
			lex.ExpectId(tok, apr.idIndividual, "IndividualID");

			if ((records.size() == 0) && (apr.idFamily == "FID") && (apr.idIndividual == "IID"))
			{
				// Get the header row
				size_t iPhenotype = 0;
				std::string phenotypeName;

				while (tok.type != tokEOF)
				{
					if (tok.type == tokEOL)
					{
						lex.NextToken(tok);
						break;
					}
					lex.ExpectId(tok, phenotypeName, "Phenotype Name");
					if (phenotypeLabelIndex.count(phenotypeName))
					{
						// alternate phenotype columns are 1 based for the user so add 1 on output.
						Fatal("Duplicate phenotype name [%s] in column %d and %d", phenotypeName.c_str(), iPhenotype + 1, phenotypeLabelIndex[phenotypeName] + 1);
					}
					phenotypeLabelIndex[phenotypeName] = iPhenotype++;
					phenotypeNames.push_back(phenotypeName);
				}

				cPhenotypes = iPhenotype;
			}
			else
			{
				// now get the alternative phenotype array and add it to records
				apr.phenotypes.clear();
				while (tok.type != tokEOF)
				{
					if (tok.type == tokEOL)
					{
						break;
					}
					limix::mfloat_t d;
					int bitBucket;
					lex.ExpectPhenotype(tok, d, bitBucket);
					apr.phenotypes.push_back(d);
				}

				// Validate we have the right number of alternative phenotypes.
				if (apr.phenotypes.size() != cPhenotypes)
				{
					if (cPhenotypes != 0)
					{
						Fatal("Alternate Phenotype file formating error in line [%d]."
							"\n                 Expected %d phenotype entries, found %d.",
							tok.line, cPhenotypes, apr.phenotypes.size());
					}

					// we didn't know how many phenotypes to expect because we
					//   didn't have labels. Create the dumy labels now that we 
					//   have read in the first line
					cPhenotypes = apr.phenotypes.size();

					char phenotypeLabel[64];
					for (size_t iPhenotypeName = 0; iPhenotypeName < cPhenotypes; ++iPhenotypeName)
					{
						sprintf(phenotypeLabel, "Phenotype_%03d", (int)(phenotypeNames.size() + 1));
						//DEBUG: - old version used "NoName" for any unnamed phenotype
						//         enable functionality until tests 'oracle' files are updated.
						char *fastLmmTesting = getenv("FastLmmTesting");
						if (fastLmmTesting != nullptr)
						{
							strcpy(phenotypeLabel, "NoName");
						}
						phenotypeLabelIndex[phenotypeLabel] = phenotypeNames.size();
						phenotypeNames.push_back(phenotypeLabel);
					}
				}

				// Validate we have a unique family:individual name and add it to the index.
				std::string key = KeyFromIdFamilyAndIdIndividual(apr.idFamily, apr.idIndividual);
				if (phenotypeIndividualIdIndex.count(key))
				{
					Fatal("Duplicate Family:Individual id [%s] found in lines %d and %d", key.c_str(), phenotypeIndividualIdIndex[key] + 1, tok.line);
				}

				phenotypeIndividualIdIndex[key] = records.size();
				records.push_back(apr);
				++cIndividuals;
				lex.NextToken(tok);      // advance to next token
			}
		}

		Verbose("              Individuals Processed: %7d", cIndividuals);
		Verbose("          Phenotypes per Individual: %7d", cPhenotypes);
	}

	size_t CPlinkAlternatePhenotypeFile::IndexFromPhenotypeLabel(const std::string& phenotypeLabel)
	{
		size_t cMatches = phenotypeLabelIndex.count(phenotypeLabel);
		if (cMatches == 0)
		{
			// this phenotype not found in map...
			Fatal("Could not locate Phenotype [%s] in Alternate Phenotype File [%s]", phenotypeLabel.c_str(), filename.c_str());
		}

		size_t phenotypeIndex = phenotypeLabelIndex[phenotypeLabel];
		return(phenotypeIndex);
	}

	/*
	 *  Return a vector of keys for Individuals that have valid phenotype values in this column
	 */
	std::vector< std::string > CPlinkAlternatePhenotypeFile::KeysForIndividualsWithValidPhenotypes(const size_t idx)
	{
		std::vector< std::string > keys;
		for (size_t iIndividual = 0; iIndividual < cIndividuals; ++iIndividual)
		{
			limix::mfloat_t val = PhenotypeValue(iIndividual, idx);
			if (val == val)
			{
				//  Valid value (not a NaN)
				keys.push_back(KeyFromIdFamilyAndIdIndividual(records[iIndividual].idFamily, records[iIndividual].idIndividual));
			}
		}

		return(keys);
	}

	/*
	 *  Return a vector of keys for Individuals that have valid phenotype values under this Phenotype Id
	 */
	std::vector< std::string > CPlinkAlternatePhenotypeFile::KeysForIndividualsWithValidPhenotypes(const std::string& idPhenotype)
	{
		return(KeysForIndividualsWithValidPhenotypes(IndexFromPhenotypeLabel(idPhenotype)));
	}

	void CPlinkAlternatePhenotypeFile::SetSelectedPhenotype(std::string& idPhenotype)
	{
		selectedPhenotype = IndexFromPhenotypeLabel(idPhenotype);
	}

	void CPlinkAlternatePhenotypeFile::SetSelectedPhenotype(size_t idx)
	{
		if (idx < cPhenotypes)
		{
			selectedPhenotype = idx;
		}
		else
		{
			Fatal("Argument for SetSelectedPhenotype() out of range.  Expected < [%d].  Found [%d]", cPhenotypes, idx);
		}
	}

	std::string& CPlinkAlternatePhenotypeFile::SelectedPhenotypeName()
	{
		return(PhenotypeName(selectedPhenotype));
	}

	std::string& CPlinkAlternatePhenotypeFile::PhenotypeName(size_t idx)
	{
		return(phenotypeNames[idx]);
	}

}// end :plink