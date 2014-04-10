#if !defined( CPlinkFile_h )
#define CPlinkFile_h
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
 *         File Name:   CPlinkFile.h
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file declare the CPlinkFile class for FastLmmC
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
	enum Genotype        // integer representation of genotype
	{
		missingGenotype = -1
		, homozygousMajor = 0
		, heterozygous = 1
		, homozygousMinor = 2
	};

	enum BedGenotype     // integer representation of genotype values in Plink's binary .BED file
	{
		bedHomozygousMinor = 0
		, bedMissingGenotype = 1
		, bedHeterozygous = 2
		, bedHomozygousMajor = 3
	};

	struct PedRecord
	{
#if 0    //bd
		FamRecord fr;
#else
		// TODO: refactor to IndividualInfo (through phenotype) for use with TFamRecord
		std::string idFamily;           // can be alpha-numeric / or numeric... :-(
		std::string idIndividual;
		std::string idPaternal;
		std::string idMaternal;
		int    sex;                // 1=male, 2=female, other=unknown
		limix::mfloat_t   phenotype;          // -9=missing, 0=missing, 1=unaffected, 2=affected, or limix::mfloat_t (anything other than 0,1,2,-9)
#endif

		std::vector<SnpNucleotides> rgSnps;        //

		PedRecord() { sex = 0; phenotype = 0.0; };
#if 0
		~PedRecord();
#endif
	};

	struct TPedRecord
	{  // TPed is the same as MapRecord for first 4 elements with the snp data for all individuals appended to each SNP
		SnpInfo snpInfo;           // the TPedRecord conains the SNP description information plus all the SNP data
		std::vector<SnpNucleotides> rgSnps;
	};

	struct DosageRecord           // information from dosage file (.DAT)
	{  // .DAT file contains two pieces of snpInfo then the dosage inforamtion for each individual
		SnpInfo  snpInfo;
		std::vector<SnpProbabilities> snpProbabilities;
	};

	// TODO:  CPlinkLexer.h Requires the SnpNucleotides definition at the top of this file...
	//          fix this later
#include "CPlinkLexer.h"      

	std::string KeyFromIdFamilyAndIdIndividual(const std::string& idFamily, const std::string& idIndividual);

	class CPlinkFile
	{
	public:
		CPlinkFile();
		CPlinkFile(const std::string& baseFilename_);
		CPlinkFile(const std::string& baseFilename_, const AlternatePhenotypeParameters& alternatePhenoptypeParameters_);
		CPlinkFile(const std::string& basefilename_, const AlternatePhenotypeParameters& alternatePhenotypeParameters_, const SnpFilterOptions& snpOptions_);
		~CPlinkFile();

		void ReadPlinkFiles(const std::string& basefilename_, PlinkFileType filetype_, const AlternatePhenotypeParameters& alternatePhenotypeParameters_, const SnpFilterOptions& snpOptions_);
		void ReadPlinkFiles(const std::string& basefilename_, PlinkFileType filetype_, const AlternatePhenotypeParameters& alternatePhenotypeParameters_);
		void ReadPlinkFiles(const std::string& basefilename_, PlinkFileType filetype_);
		void ReadPlinkFiles(PlinkFileType filetype);

		size_t   cIndividuals;
		size_t   cPhenotypes;
		size_t   cSnps;
		std::vector<std::string>  individualLabels;   // labels for individuals selected
		std::vector<std::string>  phenotypeLabels;    // labels for phenotypes selected
		std::vector<SnpInfo> rgSnpInfo;          // Meta-data Infomation for snps selected (rg means rangeof or std::vectorof)
		size_t   ind_phenotype;

		limix::mfloat_t     *snpData;                  // snpData[cIndividuals, cSnps] (column major)
		limix::mfloat_t     *phenotypeData;            // phenotypeData[cIndividuals, cPhenotypes] (column major)

		CCovariatesData *covariatesFileData;

		static inline int compareSnpInfoByPosition(const void *x, const void *y);   //if position (chromosome + basepairposition) of (x>y): 1, (x<y):-1, (x==y):0
		static inline int getBpDistance(SnpInfo &x, SnpInfo &y);       //compute BP distance. returns a large number if on diffrent chromosomes
		static inline limix::mfloat_t getGeneticDistance(SnpInfo &x, SnpInfo &y); //compute genetic distance. returns infinity if on different chromosomes

	private:
		PlinkFileType fileType;             // Natural=use PED/MAP, Binary=use BED/FAM/BIM, Transposed=use TPED/TFAM
		std::string   baseFilename;
		AlternatePhenotypeParameters  alternatePhenotypeParameters;
		CPlinkAlternatePhenotypeFile  alternatePhenotypeFile;

		SnpFilterOptions snpFilter;         // some PLink files are filtered on SNPs too.

		int      phenotypeValueType;        // 0=uninitialized, 1=affection phenotype, 2=quantitative phenotype
		size_t   cIndividualsRead;
		size_t   cSnpsRead;                 // number of SNPs read from data files

		std::vector< size_t > selectedIndividuals;  // index of individuals selected for inclusion in study
		std::vector< size_t > selectedSnps;         // index of SNPs selected for inclusion in this study

		limix::mfloat_t   *phenArray;
		limix::mfloat_t   *snpArray;
		//   limix::mfloat_t   *genotypeArray;                 //

		std::vector<PedRecord> rgPed;
		std::vector<MapRecord> rgMap;               // The Map file contains _only_ SnpInfo records
		std::vector<FamRecord> rgFam;
		//std::vector<BedRecord> rgBed;             // The Bed file is a clone of rgMap
		//std::vector<BimRecord> rgBim;
		std::vector<TPedRecord> rgTPed;
		//std::vector<TFamRecord> rgTFam;           // The TFam file is clone of rgFam
		std::vector<DosageRecord> rgDosage;         // paired with rgFam for individual info & alternate phenotype for phenotype info

		std::vector< std::vector<limix::mfloat_t> > rgBinaryGenotype;

		std::map< std::string, size_t > indexOfFamilyIdAndIndividualId;
		size_t   IndexFromIdFamilyAndIdIndividual(const std::string& idFamily, const std::string& idIndividual);
		void     AddToFamilyIdAndIndividualIdIndex(const std::string& idFamily, const std::string& idIndividual, size_t idx);

		std::map< std::string, size_t > indexOfSnpIds;
		size_t   IndexFromSnpId(const std::string& snpId);
		void     AddSnpIdToIndex(const std::string& snpId, size_t idx);

		void     ReadAltPhenotype(AlternatePhenotypeParameters& altParameters);

		void     ReadNaturalFiles();
		void     ReadPedFile();
		void     WritePedFile(const std::string& pedFile);
		void     ReadMapFile();
		void     WriteMapFile(const std::string& mapFile);
		void     ProduceFilteredDataFromNaturalFiles();

#if 0
		void     ReadBinaryFiles();
#endif
		//   void     ReadBedFile();

#if 0
		void     ReadBinaryFiles2();
		void     ReadBedFile2();

		void     ReadBinaryFiles3();
		void     ReadBedFile3(std::vector< size_t >& selectedSnps);
#endif

		void     ReadBinaryFiles4();
		void     ReadBedFile4(std::vector< size_t >& preSelectedSnps, std::vector< size_t >& finalSelectedSnps);

		//   void     WriteBedFile( const std::string& bedFile, LayoutMode layout_ );

		void     ReadFamFile();
		void     ReadFamFile(const std::string& famFile);
		void     WriteFamFile(const std::string& famFile);
		void     ReadBimFile();
		void     WriteBinFile();
#if 0
		void     ProduceFilteredDataFromBinaryFiles();
#endif
		//   void     ProduceFilteredDataFromBinaryFiles3();

		void     ReadTransposedFiles();
		void     ReadTFamFile();
		//void     WriteTFamFile( const std::string& tfamFile );
		void     ReadTPedFile();
		void     WriteTPedFile(const std::string& tmapFile);
		void     ProduceFilteredDataFromTransposedFiles();

		void     ReadDosageFiles();
		void     ProduceFilteredDataFromDosageFiles(CPlinkDatFile& datFile, CPlinkMapFile& mapFile);

		void     ComputeNaturalSnpAlleleChars();
		void     CreateNaturalColumnMajorSnpArray();
		void     CreateNaturalColumnMajorPhenArray();

		//   void     CreateBinaryColumnMajorSnpArray();
		//   void     CreateBinaryColumnMajorSnpArray3();
		void     CreateBinaryColumnMajorPhenArray();

		void     CreateTransposedColumnMajorSnpArray();
		void     CreateTransposedColumnMajorPhenArray();

		void     CreateDosageColumnMajorSnpArray(CPlinkDatFile& datFile);
		void     CreateDosageColumnMajorPhenArray();

		void     SelectIndividualsFromRgPed();
		void     SelectIndividualsFromRgFam();
		void     SelectIndividualsFromDosageFiles(CPlinkAlternatePhenotypeFile& altPhenotypeFile, CPlinkDatFile& datFile);

		void     SelectSnps(size_t cSnpsRead, SnpFilterOptions& snpFilter);
		void     SelectSnps(size_t cSnpsRead, SnpFilterOptions& snpFilter, std::vector< size_t >& selectSnps);
		void     SelectSnpsFromNaturalFiles();
		//   void     SelectSnpsFromBinaryFiles();
		//   void     SelectSnpsFromBinaryFiles3();
		//   void     SelectSnpsFromBinaryFiles3( size_t cSnpsRead, SnpFilterOptions& snpFilter, std::vector< size_t >& selectedSnps );
		void     SelectSnpsFromTransposedFiles();
		void     SelectSnpsFromDosageFiles(CPlinkDatFile& datFile);

		void     ExtractSnpsFromRgMap();
		void     ExtractSnpsFromRgMap2();      // ExtractSnpsForNaturalFiles()
		void     ExtractSnpsFromRgTPed();
		void     ExtractSnpsFromDosage(CPlinkDatFile& datFile, CPlinkMapFile& mapFile);

		void     DoInit(const std::string* basefilename, const AlternatePhenotypeParameters *alternatePhenoptypeParameters, const SnpFilterOptions *snpFilerOptions);
		limix::mfloat_t     mfloat_tFromSnpNuculeotides(char& majorAllele_, SnpNucleotides& snp);
		limix::mfloat_t     mfloat_tFromSnpProbabilities(SnpProbabilities& snpP);
		void     FreePrivateMemory();

		void     WriteDatFile(const std::string& datFilename);

		bool     FSnpHasVariation(size_t iSnp);
		//   bool     FSnpHasVariationBinaryFile( size_t iSnp );
		//   bool     FSnpHasVariation3( SnpInfo& snpInfo, std::vector< BedGenotype >& rgBedGenotypes );
		bool     FSnpHasVariation4(SnpInfo& snpInfo, std::vector< BedGenotype >& rgBedGenotypes, std::vector< size_t >& selected);
		bool     FSnpHasVariation4b(SnpInfo& snpInfo, limix::mfloat_t* genotypes);
		bool     FSnpHasVariationDosageFile(size_t iSnp);
		bool     FSnpHasVariationTransposedFile(size_t iSnp);

		void     ValidateSnpGeneticDistanceInformation(SnpInfo& snpInfo_);
		void     ValidateSnpPositionInformation(SnpInfo& snpInfo_);
	};

	// compute BP distance. returns a large number if on diffrent chromosomes
	inline int CPlinkFile::getBpDistance(SnpInfo &x, SnpInfo &y)
	{
		if (x.iChromosome == y.iChromosome)
		{
			return(x.basepairPosition - y.basepairPosition);
		}
		else if (x.iChromosome < y.iChromosome)
		{
			return(-2000000000);
		}
		else
		{
			return(2000000000);
		}
	}

	// compute genetic distance. returns infinity if on different chromosomes
	inline limix::mfloat_t CPlinkFile::getGeneticDistance(SnpInfo &x, SnpInfo &y)
	{
		if (x.iChromosome == y.iChromosome)
		{
			return(x.geneticDistance - y.geneticDistance);
		}
		else if (x.iChromosome < y.iChromosome)
		{
			return(-std::numeric_limits<limix::mfloat_t>::infinity());
		}
		else
		{
			return(std::numeric_limits<limix::mfloat_t>::infinity());
		}
	}

	inline int CPlinkFile::compareSnpInfoByPosition(const void *x, const void *y)   //if position (chromosome + basepairposition) of (x>y): 1, (x<y):-1, (x==y):0
	{
		//check the chromosome
		SnpInfo* px = (SnpInfo*)x;
		SnpInfo* py = (SnpInfo*)y;

		if (px->iChromosome > py->iChromosome)
		{
			return(1);
		}
		else if (px->iChromosome < py->iChromosome)
		{
			return(-1);
		}
		else
		{//if the chromosomes match up, compare the position
			if (px->basepairPosition > py->basepairPosition)
			{
				return(1);
			}
			else if (px->basepairPosition < py->basepairPosition)
			{
				return(-1);
			}
			else
			{
				return(0);
			}
		}
	}
}//end :plink
#define  PHENOTYPE_AFFECTION     1
#define  PHENOTYPE_QUANTITATIVE  2

#endif   // CPlinkFile_h
