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
 * CSnpFilterOptions - {SNP Filtering Options Class}
 *
 *         File Name:   CSnpFilterOptions.cpp
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   15 Mar 2012
 *     Revision Date:   15 Mar 2012
 *
 *    Module Purpose:   This file implements the CSnpFilterOptions
 *                         class for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "CPlink.h"
namespace plink {
	size_t SnpFilterOptions::IndexToFirstSnpInJob(size_t cSnpsToPartition)
	{
		cSnpsInJob = ((int)cSnpsToPartition + numberOfJobs - 1) / numberOfJobs;
		return(cSnpsInJob * thisJobIndex);
	}

	size_t SnpFilterOptions::IndexToLastSnpInJob(size_t cSnpsToPartition)
	{
		cSnpsInJob = ((int)cSnpsToPartition + numberOfJobs - 1) / numberOfJobs;
		size_t iSnpLast = (cSnpsInJob * (thisJobIndex + 1));
		iSnpLast = std::min(iSnpLast, cSnpsToPartition);
		return(iSnpLast);
	}

	void SnpFilterOptions::SetByCount(int cSnps)
	{
		if (cSnps <= 0)
		{
			Fatal("FilterOption cSnpsInJob must be a positive number.  Found [%d]", cSnps);
		}
		filterType = FilterByCount;
		cSnpsInJob = cSnps;
	}

	void SnpFilterOptions::SetByJob(int cJobs, int iJob)
	{
		if (cJobs <= 0)
		{
			Fatal("FilterOption numberOfJobs must be a positive number. Found [%d]", cJobs);
		}
		filterType = FilterByJob;
		numberOfJobs = cJobs;
		if (iJob >= numberOfJobs)
		{
			Fatal("FilterOption thisJob must be less than the numberOfJobs argument [%d]. Found [%d]", numberOfJobs, iJob);
		}
		thisJobIndex = iJob;
	}

	void SnpFilterOptions::SetByFileList(std::string& filename)
	{
		filterType = FilterByFilelist;
		extractFile = filename;
		cSnpsInJob = std::numeric_limits<int>::max();      // 
	}

	void SnpFilterOptions::SetByTopN(string& filename, int cSnps)
	{
		if (filename.size() == 0)
		{
			Fatal("FilterOption 'SetByTopN' requires a valid filename.  None found.");
		}
		if (cSnps <= 0)
		{
			Fatal("FilterOption cSnpsInJob must be a positive number.  Found [%d]", cSnps);
		}
		filterType = FilterByTopN;
		cSnpsInJob = cSnps;
		extractFile = filename;
	}

	void SnpFilterOptions::LoadSnpIdsToExtract()
	{
		if (!FFilterByFile())
		{
			return;
		}

		SnpId idSnp;
		std::map< SnpId, size_t > detectDuplicatSnpIds;

		if (filterType == FilterByFilelist)
		{
			ProgressNL("  --               Process -extract: [%s]", extractFile.c_str());
		}
		else
		{
			ProgressNL(" --            Process -extractTopK: [%s] %d", extractFile.c_str(), cSnpsInJob);
		}
		CPlinkLexer lex(extractFile);
		CToken tok;

		lex.NextToken(tok);               // prime the lexer
		while (tok.type != tokEOF)        // for each line until EOF
		{
			if (tok.type == tokEOL)        // blank line...or comment line or ???
			{
				lex.NextToken(tok);
				continue;
			}

			lex.ExpectId(tok, idSnp, "Snp ID");
			if (tok.type != tokEOL)
			{
				Fatal("Expecting <EOL> after Snp ID in file [%s] near line %d:%d.  Found [%s]", lex.Filename().c_str(), tok.line, tok.column, tok.text.c_str());
			}

			lex.NextToken(tok);      // consume the <EOL>

			if (detectDuplicatSnpIds.count(idSnp))
			{
				Fatal("Found duplicate Snp ID [%s] in file [%s] near lines %d and %d", idSnp.c_str(), lex.Filename().c_str(), detectDuplicatSnpIds[idSnp] + 1, snpIdsToExtract.size() + 1);
			}

			// Valid idSnp  Add it and move on.
			detectDuplicatSnpIds[idSnp] = snpIdsToExtract.size();     // add to the duplicate detection table
			snpIdsToExtract.push_back(idSnp);

			if (snpIdsToExtract.size() >= (size_t)cSnpsInJob)
			{
				break;
			}
		}

		ProgressNL("  --                   SnpIds found: %d", snpIdsToExtract.size());
	}
}// end :plink