#if !defined( CSnpFilterOptions_h )
#define Startup_h
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
 *         File Name:   CSnpFilterOptions.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   15 Mar 2012
 *     Revision Date:   15 Mar 2012
 *
 *    Module Purpose:   This file defines the CSnpFilterOptions
 *                         class interface for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

namespace plink {

	typedef  std::string   SnpId;

	class SnpFilterOptions              // hold all the SnpFilter related parameters
	{
	public:
		enum FilterType
		{
			FilterByNone = 0
			, FilterByCount                 // use 'n' SNPs
			, FilterByJob
			, FilterByFilelist
			, FilterByTopN
		}   filterType;
		//   bool   fUseSnpFilter;            // Do we use the Snp Filter or not?
		int    cSnpsInJob;               // how many SNPs/Genotypes to process in one pass
		int    cJobsInFile;              // the number of iterations to complete all jobs
		int    thisJobIndex;             // the iteration we are processing this invocation
		int    numberOfJobs;
		std::string extractFile;              // file with SNPid's to operate on.
		std::vector< SnpId > snpIdsToExtract; // list of SNPid's to extract

		SnpFilterOptions() { filterType = FilterByNone; cSnpsInJob = -1; cJobsInFile = -1; thisJobIndex = -1; numberOfJobs = -1; }

		size_t IndexToFirstSnpInJob(size_t cSnpsToPartition);
		size_t IndexToLastSnpInJob(size_t cSnpsToPartition);

		bool FUseSnpFilter() { return(filterType != FilterByNone); }
		bool FFilterByFile() { return((filterType == FilterByFilelist) || (filterType == FilterByTopN)); }

		void SetByCount(int cSnps);
		void SetByJob(int cJobs, int iJob);
		void SetByFileList(std::string& filename);
		void SetByTopN(std::string& filename, int cSnps);

		void LoadSnpIdsToExtract();
	};
}// end :plink
#endif   // CSnpFilterOptions_h
