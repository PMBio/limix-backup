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
 * CPlinkDatFile - {PLINK Dat File Access Class}
 *
 *         File Name:   CPlinkDatFile.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:    4 Dec 2011
 *     Revision Date:    4 Dec 2011
 *
 *    Module Purpose:   This file implements the CPlinkDatFile class 
 *                         for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

#include "CPlinkDatFile.h"
namespace plink {
	CPlinkDatFile::CPlinkDatFile(const std::string& filename_)
	{
		filename = filename_;
		pFile = nullptr;
	}

	//CPlinkDatFile::CPlinkDatFile( const std::string& filename_, size_t cIndividuals_, size_t cSnps_ ) {}

	CPlinkDatFile::~CPlinkDatFile()
	{
		if (pFile != nullptr)
		{
			fclose(pFile);
			pFile = nullptr;
		}
	}

	void CPlinkDatFile::Load()
	{
		Load(nullptr);
	}

	void CPlinkDatFile::Load(std::vector<FamRecord>* prgFam_)
	{
		SnpProbabilities         snpP;
		std::vector<SnpProbabilities> rgSnpP;
		DatRecord                dr;
		IndividualId             individualId;
		size_t                   cIndividualsExpected = prgFam_ ? prgFam_->size() : 0;

		Verbose("                  Loading .DAT file: [%s]", filename.c_str());
		CTimer timer(true);
		CToken tok;
		CPlinkLexer lex(filename);

		do { // skip leading blank lines and comments.
			lex.NextToken(tok);
		} while (tok.type == tokEOL);
		if ((tok.type == tokSymbol) && (tok.text == "SNP"))
		{
			// Header row in file
			std::string allele1Header;
			std::string allele2Header;

			lex.NextToken(tok);      // advance past "SNP" and throw away
			lex.ExpectId(tok, allele1Header, "Allele1_Header");    // advance past "A1" and throw away
			lex.ExpectId(tok, allele2Header, "Allele2_Header");    // advance past "A2" and throw away
			while ((tok.type != tokEOF) && (tok.type != tokEOL))
			{
				// Get the Family/Individual ID pairs
				lex.ExpectId(tok, individualId.idFamily, "FamilyID");
				lex.ExpectId(tok, individualId.idIndividual, "IndividualID");

				// Validate we have a unique key for this individual
				std::string key = KeyFromIdFamilyAndIdIndividual(individualId.idFamily, individualId.idIndividual);
				if (FKeyIndividualInDatFile(key))
				{
					Fatal("Duplicate FamilyId:IndividualId [%s:%s] found in header elements %d and %d",
						individualId.idFamily.c_str(),
						individualId.idIndividual.c_str(),
						keyIndividualToDatRecordIndividualIndex[key] + 1,
						keyIndividualToDatRecordIndividualIndex.size() + 1);
				}

				keyIndividualToDatRecordIndividualIndex[key] = rgIndividualIds.size();
				rgIndividualIds.push_back(individualId);
			}

			cIndividualsExpected = rgIndividualIds.size();
		}

		// No header row or bogus header row
		if (cIndividualsExpected == 0)
		{
			Fatal("Expected a header row starting with \"SNP\" in file [%s]", filename.c_str());
		}

		// We are either at the first token of the file without a header or the first token of a new line
		while (tok.type != tokEOF)        // for each line until EOF
		{
			if (tok.type == tokEOL)        // blank line...or comment line or ???
			{
				lex.NextToken(tok);
				continue;
			}

			rgSnpP.clear();
			lex.ExpectId(tok, dr.idSnp, "SnpID");
			lex.ExpectSnpAlleles(tok, dr.majorAllele, dr.minorAllele);
			while ((tok.type != tokEOF) && (tok.type != tokEOL))
			{
				lex.ExpectSnpProbabilities(tok, snpP);
				rgSnpP.push_back(snpP);
			}

			if (rgSnpP.size() != cIndividualsExpected)
			{
				Fatal("Expected SNP probability pairs for %d individuals on line %d.  Found %Id", cIndividualsExpected, tok.line, rgSnpP.size());
			}

			if (idSnpToDatRecordIndex.count(dr.idSnp))
			{
				Fatal("Duplicate SNP Id %s found in SNPs %d and %d", dr.idSnp.c_str(), idSnpToDatRecordIndex[dr.idSnp] + 1, rgDat.size() + 1);
			}

			idSnpToDatRecordIndex[dr.idSnp] = rgDat.size();
			dr.rgSnpProbabilities = rgSnpP;
			rgDat.push_back(dr);
		}

		if (rgIndividualIds.size() == 0)
		{
			/*
			 * We got here because we have not header row but a we have data.
			 *   Copy the IndividualId from the rgFam information and construct
			 *   the needed indexing support.
			 */
			for (size_t iIndividual = 0; iIndividual < prgFam_->size(); ++iIndividual)
			{
				FamRecord* pfr = &(prgFam_->at(iIndividual));
				individualId.idFamily = pfr->idFamily;
				individualId.idIndividual = pfr->idIndividual;
				std::string key = KeyFromIdFamilyAndIdIndividual(individualId.idFamily, individualId.idIndividual);
				keyIndividualToDatRecordIndividualIndex[key] = rgIndividualIds.size();
				rgIndividualIds.push_back(individualId);
			}
		}
		timer.Stop();
		timer.Report(0x02, "     Loading .DAT file elapsed time: %s");
	}

	bool CPlinkDatFile::FGetSnpInfo(const std::string& idSnp, SnpInfo& snpInfo_)
	{
		snpInfo_.Clear();

		if (idSnpToDatRecordIndex.count(idSnp) == 0)
		{
			return(false);
		}

		size_t idx = idSnpToDatRecordIndex[idSnp];
		snpInfo_.idSnp = rgDat[idx].idSnp;
		snpInfo_.majorAllele = rgDat[idx].majorAllele;
		snpInfo_.minorAllele = rgDat[idx].minorAllele;
		return(true);
	}

	bool CPlinkDatFile::FGetSnpInfo(size_t idx, SnpInfo& snpInfo_)
	{
		snpInfo_.Clear();

		if (idx >= rgDat.size())
		{
			return(false);
		}

		snpInfo_.idSnp = rgDat[idx].idSnp;
		snpInfo_.majorAllele = rgDat[idx].majorAllele;
		snpInfo_.minorAllele = rgDat[idx].minorAllele;
		return(true);
	}

	bool CPlinkDatFile::FSnpHasVariation(const std::string& idSnp)
	{
		size_t idxSnp = IdxFromIdSnp(idSnp);
		return(FSnpHasVariation(idxSnp));
	}

	bool CPlinkDatFile::FSnpHasVariation(size_t idxSnp)
	{
		DatRecord* dr = DatRecordPointer(idxSnp);
		SnpProbabilities probabilities0(-1.0, -1.0);

		for (size_t iProbability = 0; iProbability < dr->rgSnpProbabilities.size(); ++iProbability)
		{
			const SnpProbabilities& p = dr->rgSnpProbabilities[iProbability];
			if (p.probabilityOfHeterozygous != -9.0)
			{
				if (probabilities0.probabilityOfHomozygousMinor == -1.0)
				{
					probabilities0 = p;
				}
				else if (probabilities0 != p)
				{
					return(true);
				}
			}
		}
		return(false);
	}

	SnpProbabilities CPlinkDatFile::GetSnpProbabilities(const std::string& idSnp, const std::string& idIndividual, const std::string& idFamily)
	{
		size_t idxSnp = IdxFromIdSnp(idSnp);
		size_t idxIndividual = IdxFromIdFamilyAndIdIndividual(idFamily, idIndividual);
		return(rgDat[idxSnp].rgSnpProbabilities[idxIndividual]);
	}

	SnpProbabilities CPlinkDatFile::GetSnpProbabilities(const std::string& idSnp, const std::string& keyIndividual)
	{
		size_t idxSnp = IdxFromIdSnp(idSnp);
		size_t idxIndividual = IdxFromKeyIndividual(keyIndividual);
		return(rgDat[idxSnp].rgSnpProbabilities[idxIndividual]);
	}

	SnpProbabilities CPlinkDatFile::GetSnpProbabilities(size_t idxSnp, size_t idxIndividual)
	{
		return(rgDat[idxSnp].rgSnpProbabilities[idxIndividual]);
	}

	size_t CPlinkDatFile::IdxFromIdSnp(const std::string& idSnp)
	{
		if (idSnpToDatRecordIndex.count(idSnp) == 0)
		{
			Fatal("Unable to find SnpId [%s] in .DAT SNP information.", idSnp.c_str());
		}
		size_t idx = idSnpToDatRecordIndex[idSnp];
		return(idx);
	}

	size_t CPlinkDatFile::IdxFromKeyIndividual(const std::string& keyIndividual)
	{
		if (keyIndividualToDatRecordIndividualIndex.count(keyIndividual) == 0)
		{
			Fatal("Unable to find individual [%s] in .DAT genotype.", keyIndividual.c_str());
		}
		size_t idx = keyIndividualToDatRecordIndividualIndex[keyIndividual];
		return(idx);
	}

	size_t CPlinkDatFile::IdxFromIdFamilyAndIdIndividual(const std::string& idFamily, const std::string& idIndividual)
	{
		std::string key = KeyFromIdFamilyAndIdIndividual(idFamily, idIndividual);
		size_t idx = IdxFromKeyIndividual(key);
		return(idx);
	}

	bool CPlinkDatFile::FKeyIndividualInDatFile(const std::string& keyIndividual)
	{
		return(keyIndividualToDatRecordIndividualIndex.count(keyIndividual) != 0);
	}

	bool CPlinkDatFile::FIdSnpInDatFile(const std::string& idSnp)
	{
		return(idSnpToDatRecordIndex.count(idSnp) != 0);
	}

	DatRecord* CPlinkDatFile::DatRecordPointer(const std::string& idSnp)
	{
		if (idSnpToDatRecordIndex.count(idSnp) == 0)
		{
			return(nullptr);
		}

		size_t idx = idSnpToDatRecordIndex[idSnp];
		DatRecord* dr = &rgDat[idx];
		return(dr);
	}

	DatRecord* CPlinkDatFile::DatRecordPointer(size_t idxSnp)
	{
		if (idxSnp > rgDat.size())
		{
			return(nullptr);
		}

		DatRecord* dr = &rgDat[idxSnp];
		return(dr);
	}

}// end:plink