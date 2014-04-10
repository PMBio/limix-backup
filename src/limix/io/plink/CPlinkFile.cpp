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
#include "CPlinkFile.h"
namespace plink {
static limix::mfloat_t unknownOrMissing = std::numeric_limits<limix::mfloat_t>::quiet_NaN();
static limix::mfloat_t homozygousPrimaryAllele = 0.0;
static limix::mfloat_t heterozygousAllele = 1.0;
static limix::mfloat_t homozygousSecondaryAllele = 2.0;

static limix::mfloat_t mapBedGenotypeTomfloat_tAllele[4] = { 
                  homozygousSecondaryAllele,       // look-up 0
                  unknownOrMissing,                // look-up 1
                  heterozygousAllele,              // look-up 2
                  homozygousPrimaryAllele,         // look-up 3
                  };

std::string KeyFromIdFamilyAndIdIndividual( const std::string& idFamily, const std::string& idIndividual )
   {
   std::string key = idFamily + " " + idIndividual;     // change to space to avoid PLink character conflict
   return( key );
   }

int BedGenotypeFromMfloat_t( limix::mfloat_t genotype )
   {
   // extracted from PLINK BED file specification 
   //   (which is confusing so read carefully!)
   if ( genotype == homozygousPrimaryAllele )      return( 3 );
   if ( genotype == heterozygousAllele )           return( 2 );
   if ( genotype == homozygousSecondaryAllele )    return( 0 );
   if ( (genotype != genotype) || (genotype == unknownOrMissing) )   return( 1 );
   Fatal( "Unrecognized value for genotype.  Found %g", genotype );
   return( 1 );
   }

Genotype GenotypeFromSnpNucleotides( char majorAllele_, SnpNucleotides& snp )
   {
   if ( (majorAllele_ == '0') || (snp.alleles[0] == '0') )
      {
      return( missingGenotype );
      }
   char chMajor = toupper( majorAllele_ );
   char ch1 = toupper( snp.alleles[0] );
   char ch2 = toupper( snp.alleles[1] );
   if ( ch1 != ch2 )
      {
      return( heterozygous );
      }
   else if ( ch1 == chMajor )
      {
      return( homozygousMajor );
      }
   return( homozygousMinor );
   }

Genotype GenotypeFromBedByteValue( int value )
   {
   static const Genotype rgMapBedByteValueToGenotype[] = {
                  homozygousMinor,                 // look-up 0
                  missingGenotype,                 // look-up 1
                  heterozygous,                    // look-up 2
                  homozygousMajor                  // look-up 3
                  };
   Genotype genotype = rgMapBedByteValueToGenotype[ value & 0x03 ];
   return( genotype );
   }

/* ************************************************************************** */
void CPlinkFile::DoInit( const std::string* basefilename_=nullptr, 
                         const AlternatePhenotypeParameters* alternatePhenotypeParameters_=nullptr, 
                         const SnpFilterOptions* snpFilterOptions_=nullptr )
   {
   // init public scalars
   cIndividuals = 0;
   cPhenotypes = 0;
   cSnps = 0;
   ind_phenotype = 0;

   snpData = nullptr;
   phenotypeData = nullptr;
   covariatesFileData = nullptr;

   // init private scalars
   fileType = UnknownFileType;
   phenotypeValueType = 0;
   cIndividualsRead = 0;
   cSnpsRead = 0;

   phenArray = nullptr;          // used to construct the data arrays
   snpArray = nullptr;

   if ( basefilename_ ) 
      {
      baseFilename = *basefilename_;
      }
   if ( alternatePhenotypeParameters_ )
      {
      alternatePhenotypeParameters = *alternatePhenotypeParameters_;
      }
   if ( snpFilterOptions_ )
      {
      snpFilter = *snpFilterOptions_;
      }
   }

CPlinkFile::CPlinkFile()
   {
   DoInit();
   }

CPlinkFile::CPlinkFile( const std::string& basefilename )
   {
   DoInit( &basefilename );
   }

CPlinkFile::CPlinkFile( const std::string& basefilename, const AlternatePhenotypeParameters& alternatePhenotypeParameters_ )
   {
   DoInit( &basefilename, &alternatePhenotypeParameters_ );
   }

CPlinkFile::CPlinkFile( const std::string& basefilename, const AlternatePhenotypeParameters& alternatePhenotypeParameters_, const SnpFilterOptions& snpOptions_ )
   {
   DoInit( &basefilename, &alternatePhenotypeParameters_, &snpOptions_ );
   }

CPlinkFile::~CPlinkFile()
   {
   // clean-up our memory allocation
   if ( snpArray != nullptr )
      {
      delete[] snpArray;
      snpArray = nullptr;
      }
   if ( phenArray != nullptr )
      {
      delete[] phenArray;
      phenArray = nullptr;
      }
   }

size_t CPlinkFile::IndexFromIdFamilyAndIdIndividual( const std::string& idFamily, const std::string& idIndividual )
   {
   std::string key = KeyFromIdFamilyAndIdIndividual( idFamily, idIndividual );

   size_t cMatch = indexOfFamilyIdAndIndividualId.count( key );
   if ( cMatch == 0 )
      {
      // No match.  return size_t NoMatch if no match
      return( NoMatch );
      }
   size_t individualIndex = indexOfFamilyIdAndIndividualId[ key ];
   return( individualIndex );
   }

void CPlinkFile::AddToFamilyIdAndIndividualIdIndex( const std::string& idFamily, const std::string& idIndividual, size_t idx )
   {
   // Validate we have a unique family:individual name and add it to the index.
   std::string key = KeyFromIdFamilyAndIdIndividual( idFamily, idIndividual );
   if ( indexOfFamilyIdAndIndividualId.count( key ) )
      {
      Fatal( "Duplicate Family:Individual id [%s] found in lines %d and %d", key.c_str(), indexOfFamilyIdAndIndividualId[ key ]+1, idx );
      }
   indexOfFamilyIdAndIndividualId[ key ] = idx;
   }

void CPlinkFile::AddSnpIdToIndex( const std::string& snpId, size_t idx )
   {
   // Validate we have a unique SNP id and add it to the map
   if ( indexOfSnpIds.count( snpId ) )
      {
      Fatal( "Duplicate Snp Id %s found in lines %d and %d", snpId.c_str(), indexOfSnpIds[ snpId ]+1, idx );
      }
   indexOfSnpIds[ snpId ] = idx;
   }

void CPlinkFile::ReadAltPhenotype( AlternatePhenotypeParameters& altParameters )
   {
   /*
    *  Populate the alternate phenotype file information in CPlinkFile
    *  and setup the result for further processing in fastlmm
    *  Although we plan for multiple phenotypes later, we only output
    *  one at the moment, so we push the specific phenotype label and
    *  set ind_phenotype = 0, and cPhenotypes = 1 
    */
   alternatePhenotypeFile.Read( altParameters.alternatePhenotypeFilename );

   /*
    *  Make sure we have the 'right' selected phenotype information 
    *    in CPlinkAlternatePhenotypeFile too
    */
   if ( altParameters.selectedPhenotypeId.empty() )
      {
      alternatePhenotypeFile.SetSelectedPhenotype( altParameters.selectedPhenotypeIndex );
      }
   else
      {
      alternatePhenotypeFile.SetSelectedPhenotype( altParameters.selectedPhenotypeId );
      }

   /*
    *  Set the CPlinkFile phenotype values now.
    */
   //DEBUG:
   if ( phenotypeLabels.size() != 0 )
      {
      Fatal( "Expected phenotypLabels.size() to be 0, and it is not!" );
      }
   // We only use 1 phenotype at the moment.  Set it up.
   phenotypeLabels.push_back( alternatePhenotypeFile.SelectedPhenotypeName() );
   cPhenotypes = 1;
   ind_phenotype = 0;
   }

void CPlinkFile::ReadPlinkFiles( const std::string& basefilename_, PlinkFileType filetype_, const AlternatePhenotypeParameters& alternatePhenotypeParameters_, const SnpFilterOptions& snpOptions_ )
   {
   snpFilter = snpOptions_;
   ReadPlinkFiles( basefilename_, filetype_, alternatePhenotypeParameters_ );
   }

void CPlinkFile::ReadPlinkFiles( const std::string& basefilename, PlinkFileType filetype_, const AlternatePhenotypeParameters& alternatePhenotypeparameters_ )
   {
   alternatePhenotypeParameters = alternatePhenotypeparameters_;
   ReadPlinkFiles( basefilename, filetype_ );
   }

void CPlinkFile::ReadPlinkFiles( const std::string& _baseFilename, PlinkFileType filetype_ )
   {
   if ( !baseFilename.empty() )
      {
      Fatal( "PLink base file name already set to %s.", baseFilename.c_str() );
      }

   baseFilename = _baseFilename;
   ReadPlinkFiles( filetype_ );
   }

void CPlinkFile::ValidateSnpGeneticDistanceInformation( SnpInfo& snpInfo_ )
   {
   if ( (snpInfo_.iChromosome == 0) || (snpInfo_.geneticDistance < 0.0) )
      {
      Fatal( "ExcludeByGeneticDistance requires SNPs to have location information."
           "\n  SnpId %s in file [%s] does not satisfy this requirement.", 
           snpInfo_.idSnp.c_str(), baseFilename.c_str() );
      }
   }

void CPlinkFile::ValidateSnpPositionInformation( SnpInfo& snpInfo_ )
   {
   if ( (snpInfo_.iChromosome == 0) || (snpInfo_.basepairPosition <= 0) )
      {
      Fatal( "ExcludeByPosition requires SNPs to have location information."
           "\n  SnpId %s in file [%s] does not satisfy this requirement.", 
           snpInfo_.idSnp.c_str(), baseFilename.c_str() );
      }
   }

void CPlinkFile::ReadPlinkFiles( PlinkFileType filetype_ )
   {
   if ( baseFilename.empty() )
      {
      Fatal( "PLink base filename not set prior to ReadPLinkFiles." );
      }

   if ( alternatePhenotypeParameters.fUseAlternatePhenotype )
      {
      ReadAltPhenotype( alternatePhenotypeParameters );
      }
   
   ProgressNL( "   ++      Processing PLINK fileset: [%s]", baseFilename.c_str() );
   fileType = filetype_;
   if ( fileType == Natural )
      {
      ReadNaturalFiles();
      }
   else if ( fileType == Binary )
      {
      ReadBinaryFiles4();     // updated filtered version with SNC support and fseek()
#if 0
      char *pchT = getenv( "FastLmmAlternateBedReader" );
      int bedReaderToUse = 4;                 // set default value for BedReader
      if ( (pchT != nullptr) )
         {
         bedReaderToUse = atoi( pchT );
         }
      switch( bedReaderToUse )
         {
      case 2:
         ReadBinaryFiles2();     // partial file read using fseek() support
         break;
      case 3:
         ReadBinaryFiles3();     // updated filtered version with SNC support
         break;
      case 4:
         ReadBinaryFiles4();     // updated filtered version with SNC support and fseek()
         break;
      default:
         ReadBinaryFiles3();     
         break;
         }
#endif
      }
   else if ( fileType == Transposed )
      { 
      ReadTransposedFiles();
      }
   else if ( fileType == Dosage )
      {
      ReadDosageFiles();
      }
   else
      {
      Fatal( "Unrecognized PLink file type.  [%d]", fileType );
      }

   /*
    *  First validate the file if we need to have it in order.
    */
   if ( FExcludeByGeneticDistance() )
      {
      ValidateSnpGeneticDistanceInformation( rgSnpInfo[ 0 ] );
      for( size_t iSnp=1; iSnp<cSnps; ++iSnp )
         {
         ValidateSnpGeneticDistanceInformation( rgSnpInfo[ iSnp ] );
         if ( getGeneticDistance( rgSnpInfo[ iSnp ], rgSnpInfo[ iSnp - 1 ] ) < 0 )
            {
            Fatal( "ExcludeByGeneticDistance requires SNPs be in ascending order."
                 "\n  SnpId %s & %s in file [%s] do not satisfy this requirement.", 
                 rgSnpInfo[iSnp].idSnp.c_str(), rgSnpInfo[iSnp-1].idSnp.c_str(), baseFilename.c_str() );
            }
         }
      }
   if ( FExcludeByPosition() )
      {
      ValidateSnpPositionInformation( rgSnpInfo[ 0 ] );
      for( size_t iSnp=1; iSnp<cSnps; ++iSnp )
         {
         ValidateSnpPositionInformation( rgSnpInfo[ iSnp ] );
         if ( getBpDistance( rgSnpInfo[ iSnp ], rgSnpInfo[ iSnp - 1 ] ) < 0 )
            {
            Fatal( "ExcludeByPosition requires SNPs be in ascending order."
                 "\n  SnpId %s & %s in file [%s] do not satisfy this requirement.", 
                 rgSnpInfo[iSnp].idSnp.c_str(), rgSnpInfo[iSnp-1].idSnp.c_str(), baseFilename.c_str() );
            }
         }
      }

   ProgressNL( "     Number of Individuals Selected: %7d", cIndividuals );
   ProgressNL( "    Number of Phenotypes/Individual: %7d", cPhenotypes );
   ProgressNL( "          Number of SNPs/Individual: %7d", cSnpsRead );
   ProgressNL( "     Number of SNPs/Individual Used: %7d", cSnps );
   ProgressNL( "   --  End Processing PLINK fileset: [%s]", baseFilename.c_str() );
   FreePrivateMemory();
   }

void CPlinkFile::ProduceFilteredDataFromNaturalFiles()
   {
   // Data has all been read in.  Convert to appropriate form.
   ExtractSnpsFromRgMap2();

   // Create output arrays adjusting for missing phenotype data.
   CreateNaturalColumnMajorSnpArray();
   CreateNaturalColumnMajorPhenArray();
   snpData = snpArray;
   phenotypeData = phenArray;
   snpArray = nullptr;
   phenArray = nullptr;
   }

#if 0
void CPlinkFile::ProduceFilteredDataFromBinaryFiles3()
   {
   // Data has all been read in.  Convert to appropriate form for processing.
   ExtractSnpsFromRgMap();

   // Create output arrays adjusting for missing phenotype data.
   CreateBinaryColumnMajorSnpArray3(); 
   CreateBinaryColumnMajorPhenArray(); 
   snpData = snpArray; 
   phenotypeData = phenArray;
   snpArray = nullptr;
   phenArray = nullptr;
   }
#endif

void CPlinkFile::ProduceFilteredDataFromTransposedFiles()
   {
   ExtractSnpsFromRgTPed();

   // Create output arrays adjusting for missing phenotype data.
   CreateTransposedColumnMajorSnpArray();
   CreateTransposedColumnMajorPhenArray();
   snpData = snpArray;
   phenotypeData = phenArray;
   snpArray = nullptr;
   phenArray = nullptr;
   }

void CPlinkFile::WriteDatFile( const std::string& datFileName )
   {
   FILE *pFile;

   if ( datFileName.empty() )
      {
      Fatal( "No .DAT output filename to open " );
      }

   pFile = fopen( datFileName.c_str(), "wt" );          // write ascii

   if ( !pFile )
      {
      std::string fullPath = FullPath( datFileName );
      Fatal( "Cannot open output file [%s].  \n  CRT Error %d: %s", fullPath.c_str(), errno, strerror( errno ) );
      }

   // for each snp, write the 'dosage' for each individual
   for ( size_t iSnp=0; iSnp<rgMap.size(); ++iSnp )
      {
      SnpInfo& snpInfo = rgMap[iSnp].snpInfo;
      fprintf( pFile, "%s  %c %c", snpInfo.idSnp.c_str(), snpInfo.minorAllele, snpInfo.majorAllele );
      for( size_t iIndividual=0; iIndividual<rgPed.size(); ++iIndividual )
         {
         limix::mfloat_t percentHomozygousMinorAllele;
         limix::mfloat_t percentHeterozygousAllele;
         limix::mfloat_t r = limix::mfloat_tFromSnpNuculeotides( snpInfo.majorAllele, rgPed[iIndividual].rgSnps[iSnp] );
         if ( r == homozygousPrimaryAllele )
            {
            percentHomozygousMinorAllele = 0.0;
            percentHeterozygousAllele = 0.0;
            }
         else if ( r == heterozygousAllele )
            {
            percentHomozygousMinorAllele = 0.0;
            percentHeterozygousAllele = 1.0;
            }
         else if ( r == homozygousSecondaryAllele )
            {
            percentHomozygousMinorAllele = 1.0;
            percentHeterozygousAllele = 0.0;
            }
         else if ( r != r )
            {
            percentHomozygousMinorAllele = -9.0;
            percentHeterozygousAllele = -9.0;
            }
         else
            {
            Fatal( "Cannot map limix::mfloat_tFromSnpNucleotides() properly." );
            }
         fprintf( pFile, "  %5.2f %5.2f", percentHomozygousMinorAllele, percentHeterozygousAllele );
         }
      fprintf( pFile, "\n" );
      }
   fclose( pFile );
   }

void CPlinkFile::ReadNaturalFiles()
   {
   CTimer timer( true );
   ReadMapFile();
   ReadPedFile();
   timer.Stop();
   timer.Report( 0x02, "      ReadNaturalFiles elapsed time: %s" );

   size_t cMapSnps = rgMap.size();
   size_t cPedSnps = rgPed[0].rgSnps.size();
   if ( cMapSnps != cPedSnps )
      {
      Fatal( "Inconsistent Snp data in PLINK file set.  MAP has %d snps and PED has %d snps", cMapSnps, cPedSnps );
      }

   ComputeNaturalSnpAlleleChars();

   char* t = getenv( "FastLmmWriteDatFile" );
   if ( t != nullptr )
      {
      std::string filename = baseFilename + ".DAT";
      WriteDatFile( filename );
      }

   // ProduceFilteredData* routines need cIndividualsRead, cPhenotypes, cSnps set prior to entry
   //DEBUG:  Assert Phenotype info is already setup
   if ( (cPhenotypes != 1) || (ind_phenotype != 0) || (phenotypeLabels.size() != 1) )
      {
      Fatal( "Phenotype info not properly setup, expected 1, 0, 1 and found %d, %d, %d", cPhenotypes, ind_phenotype, phenotypeLabels.size() );
      }

   cIndividualsRead = rgPed.size();
   SelectIndividualsFromRgPed();

   cSnpsRead = rgPed[0].rgSnps.size();
   SelectSnpsFromNaturalFiles();
   ProduceFilteredDataFromNaturalFiles();
   }

void CPlinkFile::SelectSnpsFromNaturalFiles()
   {
   // Select the SNPs to propagate based on filtering
   //   and the individuals that are in the study sample
   //   If there is no genotype variation within the SNP
   //   in the study sample, remove the SNP from the study
   switch( snpFilter.filterType )
      {
   case SnpFilterOptions::FilterByNone:
      // No SNP filter applied.  Filter _ONLY_ 'constant SNPs'
      selectedSnps.reserve( cSnpsRead );
      for ( size_t iSnp=0; iSnp<cSnpsRead; ++iSnp )
         {
         if ( FSnpHasVariation( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ iSnp ].snpInfo.idSnp.c_str() );
            }
         }
      break;
   case SnpFilterOptions::FilterByJob:
      {
      // No extract file, so it must be a sequential subset 
      //   created from 'Job' and JobNumber
      size_t firstSnp = snpFilter.IndexToFirstSnpInJob( cSnpsRead );
      size_t lastSnp = snpFilter.IndexToLastSnpInJob( cSnpsRead );
      selectedSnps.reserve( lastSnp - firstSnp );

      for( size_t iSnp=firstSnp; iSnp<lastSnp; ++iSnp )
         {
         if ( FSnpHasVariation( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ iSnp ].snpInfo.idSnp.c_str() );
            }
         }
      }
      break;
   case SnpFilterOptions::FilterByFilelist:
   case SnpFilterOptions::FilterByTopN:
      // match the snps in snpFilter.snpsToProcess against the snps
      //   we have the map of SnpIds to index in indexOfSnpIds
      //   we have the list of SnpIds we want in snpFilter.snpIdsToExtract
      //   get the indexes put into selectedSnps (and then sort?) to be in original order
      selectedSnps.reserve( snpFilter.snpIdsToExtract.size() );
      for ( size_t i=0; i<snpFilter.snpIdsToExtract.size(); ++i )
         {
         if ( indexOfSnpIds.count( snpFilter.snpIdsToExtract[i] ) == 0 )
            {
            Warn( "SNP not found.  Cannot extract SNP information for [%s].  Skipping...", snpFilter.snpIdsToExtract[i].c_str() );
            }
         else
            {
            size_t iSnp = indexOfSnpIds[ snpFilter.snpIdsToExtract[i] ];
            if ( FSnpHasVariation( iSnp ) )
               {
               selectedSnps.push_back( iSnp );
               }
            else
               {
               Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ i ].snpInfo.idSnp.c_str() );
               }
            }
         }
      sort( selectedSnps.begin(), selectedSnps.end() );
      break;
   default:
      Fatal( "Unexpected snpFilter type.  %d", snpFilter.filterType );
      }

   cSnps = selectedSnps.size();
   if ( cSnps == 0 )
      {
      Fatal( "No SNPs passed filter criteria." );
      }
   }

#if 0
/*
 *
 */
void CPlinkFile::ReadBinaryFiles2()
   {
   CTimer timer( false );

   /*
    *  Read in the ASCII files that are not addressable using fseek()
    */
   timer.Start( true );
   ReadFamFile();             // idFamily, idIndividual, idFather, idMother, sex, phenotype
   timer.Stop();
   timer.Report( 0x02, "        ReadFamFile()* elapsed time: %s" );

   timer.Start( true );
   ReadBimFile();             // SnpInfo - Chromosome, idSnp, Genetic Distance, Basepair Position, alleleMinor, alleleMajor
   timer.Stop();
   timer.Report( 0x02, "        ReadBimFile()* elapsed time: %s" );

   /*
    *  Make sure all the 'selection criteria is setup so we can read 
    *    from the binary file using only one 'line' of information at
    *    a time and not needing to have two copies of all the stuff
    *    to filter and sort it out.
    */
   // ProduceFilteredData* routines need cIndividualsRead, cPhenotypes, cSnps set prior to entry
   //DEBUG:  Assert Phenotype info is already setup
   if ( (cPhenotypes != 1) || (ind_phenotype != 0) || (phenotypeLabels.size() != 1) )
      {
      Fatal( "Phenotype info not properly setup, expected 1, 0, 1 and found %d, %d, %d", cPhenotypes, ind_phenotype, phenotypeLabels.size() );
      }

   cIndividualsRead = rgFam.size();
   SelectIndividualsFromRgFam();
   cSnpsRead = rgMap.size();

   timer.Start( true );
   SelectSnps( cSnpsRead, snpFilter );    // populate selectedSnps<> create list of applies filter to SNP selection and sets cSnps
   ExtractSnpsFromRgMap();                // need to copy them over for later use too.
   CreateBinaryColumnMajorPhenArray();    // copy Phenotype info to array rather than std::vector
   timer.Stop();
   timer.Report( 0x02, "  SNP & Phenotype info elapsed time: %s" );

   // Due to the binary nature of the bed file,
   //   we can create the binary array while 
   //   reading the file.
   timer.Start( true );
   ReadBedFile2();
   timer.Stop();
   timer.Report( 0x02, "        ReadBedFile()* elapsed time: %s" );

   snpData = snpArray; 
   phenotypeData = phenArray;
   snpArray = nullptr;
   phenArray = nullptr;
   }
#endif

#if 0
/*
 *
 */
void CPlinkFile::ReadBinaryFiles3()
   {
   CTimer timer( false );

   /*
    *  Read in the ASCII files that are not addressable using fseek()
    */
   timer.Start( true );
   ReadFamFile();             // idFamily, idIndividual, idFather, idMother, sex, phenotype
   timer.Stop();
   timer.Report( 0x02, "        ReadFamFile()* elapsed time: %s" );

   timer.Start( true );
   ReadBimFile();             // SnpInfo - Chromosome, idSnp, Genetic Distance, Basepair Position, alleleMinor, alleleMajor
   timer.Stop();
   timer.Report( 0x02, "        ReadBimFile()* elapsed time: %s" );

   timer.Start( true );
   ReadBedFile();
   timer.Stop();
   timer.Report( 0x02, "         ReadBedFile() elapsed time: %s" );

   // ProduceFilteredData* routines need cIndividualsRead, cPhenotypes, cSnps set prior to entry
   //DEBUG:  Assert Phenotype info is already setup
   if ( (cPhenotypes != 1) || (ind_phenotype != 0) || (phenotypeLabels.size() != 1) )
      {
      Fatal( "Phenotype info not properly setup, expected 1, 0, 1 and found %d, %d, %d", cPhenotypes, ind_phenotype, phenotypeLabels.size() );
      }

   cIndividualsRead = rgFam.size();
   cSnpsRead = rgMap.size();
   SelectIndividualsFromRgFam();
   SelectSnpsFromBinaryFiles3();    // MUST have individuals already selected to filter SNCs

   timer.Start( true );
   ProduceFilteredDataFromBinaryFiles3();
   timer.Stop();
   timer.Report( 0x02, "   ProduceFilteredData elapsed time: %s" );
   }
#endif

void CPlinkFile::ReadBinaryFiles4()
   {
   CTimer timer( false );
   CTimer timerReadBinaryFiles4( true );

   /*
    *  Read in the ASCII files that are not addressable using fseek()
    */
   timer.Start( true );
   ReadFamFile();             // idFamily, idIndividual, idFather, idMother, sex, phenotype
   timer.Stop();
   timer.Report( 0x02, "        ReadFamFile()* elapsed time: %s" );

   timer.Start( true );
   ReadBimFile();             // SnpInfo - Chromosome, idSnp, Genetic Distance, Basepair Position, alleleMinor, alleleMajor
   timer.Stop();
   timer.Report( 0x02, "        ReadBimFile()* elapsed time: %s" );

   /*
    *  Make sure all the 'selection criteria is setup so we can read 
    *    from the binary file using only one 'line' of information at
    *    a time and not needing to have two copies of all the stuff
    *    to filter and sort it out.
    */
   // ProduceFilteredData* routines need cIndividualsRead, cPhenotypes, cSnps set prior to entry
   //DEBUG:  Assert Phenotype info is already setup
   if ( (cPhenotypes != 1) || (ind_phenotype != 0) || (phenotypeLabels.size() != 1) )
      {
      Fatal( "Phenotype info not properly setup, expected 1, 0, 1 and found %d, %d, %d", cPhenotypes, ind_phenotype, phenotypeLabels.size() );
      }

   // Deal with fields about Individuals
   cIndividualsRead = rgFam.size();       // make we have valid phenotypes and create the phenotype array
   SelectIndividualsFromRgFam();
   CreateBinaryColumnMajorPhenArray();    // copy Phenotype info to array rather than std::vector.  ONLY needs SelectedIndividuals

   timer.Start( true );
   std::vector< size_t > preSelectedSnps;
   cSnpsRead = rgMap.size();              // make sure we have the 'right' SNPs to process

   SelectSnps( cSnpsRead, snpFilter, preSelectedSnps );    // populate preSelectedSnps<> create list of applies filter to SNP selection and sets cSnps
   timer.Stop();
   timer.Report( 0x02, "  SNP & Phenotype info elapsed time: %s" );

   // Due to the binary nature of the bed file,
   //   we can create the binary array while 
   //   reading the file.
   timer.Start( true );
   ReadBedFile4( preSelectedSnps, selectedSnps );
   timer.Stop();
   timer.Report( 0x02, "       ReadBedFile4()* elapsed time: %s" );

   snpData = snpArray; 
   phenotypeData = phenArray;
   snpArray = nullptr;
   phenArray = nullptr;

   timerReadBinaryFiles4.Stop();
   timerReadBinaryFiles4.Report( -1, "   ReadBinaryFiles4()* elapsed time: %s" );
   }

void CPlinkFile::ReadBimFile()
   {
   ReadMapFile();
   }

#if 0
void CPlinkFile::ReadBedFile()
   {
   // Read the binary PLINK file
   std::string filename = baseFilename + ".bed";
   Verbose( "              Processing PLink file: [%s]", filename.c_str() );

   size_t cIndividualsInFile = rgFam.size();
   size_t cSnpsInFile = rgMap.size();

   CBedFile bf( filename, cIndividualsInFile, cSnpsInFile );
   rgBinaryGenotype = std::vector< std::vector<limix::mfloat_t> >(cIndividualsInFile, std::vector<limix::mfloat_t>(cSnpsInFile));

   unsigned int genotypeByte;

   if ( bf.layout == GroupGenotypesByIndividual )
      {
      // read in Individual-major order (row major order) (all genotypes for one individual together)
      for ( size_t iIndividual=0; iIndividual<cIndividualsInFile; ++iIndividual )
         { 
         // Get all the SNPS for this individual
         for(size_t iSnp = 0; iSnp < cSnpsInFile; )
            {
            genotypeByte = bf.NextChar();

            // manually unrolled loop
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[( genotypeByte       & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 2) & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 4) & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 6) & 0x03)];
            }
         }
      }
   else  // else GroupGenotypesBySnp 
      {
      // read in SNP-major order (all genotypes for one SNP together)
      for ( size_t iSnp=0; iSnp<cSnpsInFile; ++iSnp )
         { 
         // Get all this SNP for all individuals
         for (size_t iIndividual = 0; iIndividual < cIndividualsInFile; )
            {
            genotypeByte = bf.NextChar();

            // manually unrolled loop
            if ( iIndividual < cIndividualsInFile ) rgBinaryGenotype[iIndividual++][iSnp] = mapBedGenotypeTolimix::mfloat_tAllele[( genotypeByte       & 0x03)];
            if ( iIndividual < cIndividualsInFile ) rgBinaryGenotype[iIndividual++][iSnp] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 2) & 0x03)];
            if ( iIndividual < cIndividualsInFile ) rgBinaryGenotype[iIndividual++][iSnp] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 4) & 0x03)];
            if ( iIndividual < cIndividualsInFile ) rgBinaryGenotype[iIndividual++][iSnp] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 6) & 0x03)];
            }
         }
      }

   if ( fCreateReadValidationFiles )         // test global to write validation file 
      {
      std::string s = filename + ".validate.txt";
      WriteBedFile( s, bf.layout );
      }
   }
#endif

#if 0
void CPlinkFile::ReadBedFile2()
   {
   // Read the binary PLINK file and produce the output
   std::string filename = baseFilename + ".bed";
   Verbose( "              Processing PLink file: [%s]", filename.c_str() );

   size_t cIndividualsInFile = rgFam.size();
   size_t cSnpsInFile = rgMap.size();

   if ( cIndividuals != individualLabels.size() ) Fatal( "broken individual counts in ReadBedFile2" );
   if ( snpFilter.fUseSnpFilter && (cSnps != selectedSnps.size())) Fatal( "broken SNP count in ReadBedFile2" );
   if ( !snpFilter.fUseSnpFilter && (cSnps != cSnpsRead) ) Fatal( "broken SNP count in ReadBedFile2-b" );

   CBedFile bf( filename, cIndividualsInFile, cSnpsInFile );
   
   std::vector< limix::mfloat_t > genotype = std::vector< limix::mfloat_t >(cIndividualsInFile);
   size_t cArrayElements = cIndividuals * cSnps;
   snpArray = new limix::mfloat_t[ cArrayElements ];

   unsigned int genotypeByte;

   if ( bf.layout == RowMajor )
      {
      // read in row major order ((all SNPs for one individual together))
      for ( size_t iIndividual=0; iIndividual<cIndividualsInFile; ++iIndividual )
         { 
         // Get all the SNPS for this individual
         for (size_t iSnp = 0; iSnp < cSnpsInFile; )
            {
            genotypeByte = bf.NextChar();

            // manually unrolled loop
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[( genotypeByte       & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 2) & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 4) & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 6) & 0x03)];
            }
         }
      }
   else
      {
      // read in column major order (all individuals for one SNP together)
      size_t cbOld = (cIndividualsInFile + 3) / 4;       // round to number of bytes
      size_t cb = bf.cbStride;
      if ( cbOld != cb ) Fatal( "bad cbStride assumptions." );

      BYTE *rgb = new BYTE[cb];

      // Get the specified SNPs
      for ( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         size_t selectedSnp = iSnp;
         if ( snpFilter.fUseSnpFilter )
            {
            selectedSnp = selectedSnps[iSnp ];
            }

         // Get this SNP for all individuals and decode the compressed
         //   form into a std::vector so we can more easily use it
         bf.ReadLine( rgb, selectedSnp );

         size_t iIndividual = 0;
         for ( size_t ib = 0; ib < cb; ++ib )
            {
            genotypeByte = rgb[ ib ];

            // manually unrolled loop
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[( genotypeByte       & 0x03)];
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 2) & 0x03)];
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 4) & 0x03)];
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 6) & 0x03)];
            }

         // Copy the data to the right output location for the array vs. the std::vector
         for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
            {
            size_t selectedIndividual = selectedIndividuals[ iIndividual ];
            size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnp );
            snpArray[ index ] = genotype[ selectedIndividual ];
            }
         }

      delete [] rgb;
      }

   // Reading a partial file precludes our ability to re-create the files...
   if ( fCreateReadValidationFiles )         // test global to write validation file 
      {
      std::string s = filename + ".validate.txt";
      Warn( "Read partial .BED file.  Cannot write validation BED file [%s]", s.c_str());
      }
   }
#endif

void CPlinkFile::ReadBedFile4( std::vector< size_t >& preSelectedSnps, std::vector< size_t >& finalSelectedSnps )
   {
   // Read the binary PLINK file and produce the output
   std::string filename = baseFilename + ".bed";
   Verbose( "              Processing PLink file: [%s]", filename.c_str() );

   size_t cIndividualsInFile = rgFam.size();
   size_t cSnpsInFile = rgMap.size();

   if ( cIndividuals != individualLabels.size() ) Fatal( "broken individual counts in ReadBedFile4" );
   if ( cIndividuals != selectedIndividuals.size() ) Fatal( "broken individual counts in ReadBedFile4-b" );
   if ( snpFilter.FUseSnpFilter() && (cSnps != preSelectedSnps.size())) Fatal( "broken SNP count in ReadBedFile4" );
   if ( !snpFilter.FUseSnpFilter() && (cSnps != cSnpsRead) ) Fatal( "broken SNP count in ReadBedFile4-b" );

   CBedFile bf( filename, cIndividualsInFile, cSnpsInFile );
   
   std::vector< limix::mfloat_t > genotype = std::vector< limix::mfloat_t >(cIndividualsInFile);

   size_t cArrayElements = cIndividuals * cSnps;      // This over allocates when SNC are removed
   snpArray = new limix::mfloat_t[ cArrayElements ];

   unsigned int genotypeByte;

   if ( bf.layout == GroupGenotypesByIndividual )
      {
      Fatal( "SNC detection in .BED files grouped by individual is not currently supported" );
      Warn( ".BED files organized as SNPs per Individual are less performant."
            "\n  Consider files organized as Individuals per SNP." );

      //TODO:  need test files
      // read in the genotypes of all SNPs for one individual 
      size_t cbOld = (cSnpsInFile + 3) / 4;           // round to number of bytes
      size_t cb = bf.cbStride;
      if ( cbOld != cb ) Fatal( "bad cbStride assumptions." );

      BYTE *rgb = new BYTE[ cb ];
      std::vector< BedGenotype > bedGenotypes = std::vector< BedGenotype >(cIndividualsInFile);

      for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
         {
         size_t selectedIndividual = selectedIndividuals[ iIndividual ];

         bf.ReadLine( rgb, selectedIndividual );

         // Get all the SNPS for this individual
         size_t iSnp = 0;
         for ( size_t ib = 0; ib < cb; ++ib )
            {
            genotypeByte = rgb[ ib ];

            // manually unrolled loop
#if 0
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[( genotypeByte       & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 2) & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 4) & 0x03)];
            if ( iSnp < cSnpsInFile ) rgBinaryGenotype[iIndividual][iSnp++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 6) & 0x03)];
#endif
            if ( iSnp < cSnpsInFile ) bedGenotypes[iSnp++] = (BedGenotype)( genotypeByte       & 0x03);
            if ( iSnp < cSnpsInFile ) bedGenotypes[iSnp++] = (BedGenotype)((genotypeByte >> 2) & 0x03);
            if ( iSnp < cSnpsInFile ) bedGenotypes[iSnp++] = (BedGenotype)((genotypeByte >> 4) & 0x03);
            if ( iSnp < cSnpsInFile ) bedGenotypes[iSnp++] = (BedGenotype)((genotypeByte >> 6) & 0x03);
            }

         // Now copy these genotype values to the output array
         for ( iSnp = 0; iSnp < cSnps; ++iSnp )
            {
            size_t selectedSnp = preSelectedSnps[ iSnp ];
            size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnp );
            snpArray[ index ] = mapBedGenotypeTolimix::mfloat_tAllele[ bedGenotypes[ selectedSnp ] ];
            }
         }
      //  We do not yet support SNC detection with the random access.  
      //TODO:  Add SNC detection and repair
      limix::mfloat_t *snpArraySrc = snpArray;
      limix::mfloat_t *snpArrayDest = snpArray;
      for( size_t iSnp = 0; iSnp < cSnps; ++iSnp )
         {
         size_t selectedSnp = selectedSnps[ iSnp ];
         SnpInfo &snpInfo = rgMap[ selectedSnp ].snpInfo;
         if ( FSnpHasVariation4b( snpInfo, snpArraySrc ) )
            {
            if ( snpArraySrc != snpArrayDest )
               {
               memmove( snpArrayDest, snpArraySrc, (cIndividuals * sizeof( limix::mfloat_t )) );
               }
            snpArrayDest += cIndividuals;
            finalSelectedSnps.push_back( selectedSnp );
            rgSnpInfo.push_back( snpInfo );
            }
         else
            {
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", snpInfo.idSnp.c_str() );
            }
         snpArraySrc += cIndividuals;
         }

      delete [] rgb;
      }
   else if ( bf.layout == GroupGenotypesBySnp )
      {
      // read in the genotypes of all individuals for one SNP
      size_t cbOld = (cIndividualsInFile + 3) / 4;       // round to number of bytes
      size_t cb = bf.cbStride;
      if ( cbOld != cb ) Fatal( "bad cbStride assumptions." );

      BYTE *rgb = new BYTE[cb];
      std::vector< BedGenotype > bedGenotypes = std::vector< BedGenotype >(cIndividualsInFile);

      // Get the specified SNPs
      for ( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         size_t selectedSnp = preSelectedSnps[ iSnp ];

         // For this SNP get all the genotype values for all individuals
         //    and decode the compressed form into a std::vector so we can more 
         //    easily use it
         bf.ReadLine( rgb, selectedSnp );

         size_t iIndividual = 0;
         for ( size_t ib = 0; ib < cb; ++ib )
            {
            genotypeByte = rgb[ ib ];

            // manually unrolled loop
            if ( iIndividual < cIndividualsInFile ) bedGenotypes[iIndividual++] = (BedGenotype)( genotypeByte       & 0x03);
            if ( iIndividual < cIndividualsInFile ) bedGenotypes[iIndividual++] = (BedGenotype)((genotypeByte >> 2) & 0x03);
            if ( iIndividual < cIndividualsInFile ) bedGenotypes[iIndividual++] = (BedGenotype)((genotypeByte >> 4) & 0x03);
            if ( iIndividual < cIndividualsInFile ) bedGenotypes[iIndividual++] = (BedGenotype)((genotypeByte >> 6) & 0x03);
#if 0
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[( genotypeByte       & 0x03)];
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 2) & 0x03)];
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 4) & 0x03)];
            if ( iIndividual < cIndividualsInFile ) genotype[iIndividual++] = mapBedGenotypeTolimix::mfloat_tAllele[((genotypeByte >> 6) & 0x03)];
#endif
            }

         // Validate there is variation across all the individual genotype values for this SNP
         if ( FSnpHasVariation4( rgMap[selectedSnp].snpInfo, bedGenotypes, selectedIndividuals ) )
            {
            // copy the data to the output array and translate to limix::mfloat_tAllele along the way
            size_t iSnpOut = finalSelectedSnps.size();
            for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
               {
               size_t selectedIndividual = selectedIndividuals[ iIndividual ];
               size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnpOut );
               snpArray[ index ] = mapBedGenotypeTolimix::mfloat_tAllele[ bedGenotypes[ selectedIndividual ] ];
               }
            finalSelectedSnps.push_back( selectedSnp );
            rgSnpInfo.push_back( rgMap[ selectedSnp ].snpInfo );
            }
         else
            {
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ selectedSnp ].snpInfo.idSnp.c_str() );
            }
         }

      delete [] rgb;
      }
   else
      {
      Fatal( "Unrecognized layout type in .BED file [%s].  Found %d", filename.c_str(), bf.layout );
      }

   // Set the cSnps value and fatal if we have filtered all the SNPs out
   this->cSnps = rgSnpInfo.size();
   if ( this->cSnps == 0 )
      {
      Fatal( "No SNPs passed filter criteria." );
      }

   // Reading a partial file precludes our ability to re-create the files...
   if ( fCreateReadValidationFiles )         // test global to write validation file 
      {
      std::string s = filename + ".validate.txt";
      Warn( "Read partial .BED file.  Cannot write validation BED file [%s]", s.c_str());
      }
   }

#if 0
   void CPlinkFile::SelectSnpsFromBinaryFiles3()
   {
   // Select the SNPs to propagate based on filtering
   //   and the individuals that are in the study sample
   //   If there is no genotype variation within the SNP
   //   in the study sample, remove the SNP from the study
   if ( !snpFilter.fUseSnpFilter )
      {
      // No SNP filter applied.  Use all the SNPs
      selectedSnps.reserve( cSnpsRead );
      for ( size_t iSnp=0; iSnp<cSnpsRead; ++iSnp )
         {
         if ( FSnpHasVariationBinaryFile( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ iSnp ].snpInfo.idSnp.c_str() );
            }
         }
      cSnps = selectedSnps.size();
      }
   else if ( snpFilter.extractFile.empty() )
      {
      // No extract file, so it must be a sequential subset 
      //   created from 'Job' and JobNumber
      snpFilter.sizeOfJob = ((int)cSnpsRead + snpFilter.numberOfJobs - 1) / snpFilter.numberOfJobs;    // round up for the size 
      size_t firstSnp = snpFilter.sizeOfJob * snpFilter.thisJobIndex;
      size_t lastSnp = min( (firstSnp + snpFilter.sizeOfJob), cSnpsRead );
      selectedSnps.reserve( lastSnp - firstSnp );

      for( size_t iSnp=firstSnp; iSnp<lastSnp; ++iSnp )
         {
         if ( FSnpHasVariationBinaryFile( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ iSnp ].snpInfo.idSnp.c_str() );
            }
         }
      cSnps = selectedSnps.size();
      }
   else
      {
      // match the snps in snpFilter.snpsToProcess against the snps
      //   we have the map of SnpIds to index in indexOfSnpIds
      //   we have the list of SnpIds we want in snpFilter.snpsToProcess
      //   get the indexes put into selectedSnps (and then sort?)
      selectedSnps.reserve( snpFilter.snpsToProcess.size() );
      for ( size_t i=0; i<snpFilter.snpsToProcess.size(); ++i )
         {
         if ( indexOfSnpIds.count( snpFilter.snpsToProcess[i] ) == 0 )
            {
            Warn( "SNP not found.  Cannot extract SNP information for [%s].  Skipping...", snpFilter.snpsToProcess[i].c_str() );
            }
         else
            {
            size_t iSnp = indexOfSnpIds[ snpFilter.snpsToProcess[i] ];
            if ( FSnpHasVariationBinaryFile( iSnp ) )
               {
               selectedSnps.push_back( iSnp );
               }
            else
               {
               Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ i ].snpInfo.idSnp.c_str() );
               }
            }
         }
      sort( selectedSnps.begin(), selectedSnps.end() );
      cSnps = selectedSnps.size();
      }
   if ( cSnps == 0 )
      {
      Fatal( "No SNPs passed filter criteria." );
      }
   }
#endif

#if 0
   void CPlinkFile::WriteBedFile( const std::string& bedFile, LayoutMode layout_ )
   {
   FILE *pFile;

   if ( bedFile.empty() )
      {
      Fatal( "No output filename to open " );
      }

   pFile = fopen( bedFile.c_str(), "wb" );          // write binary

   if ( !pFile )
      {
      Fatal( "Cannot open output file [%s].  \n  CRT Error %d: %s", FullPath( bedFile ).c_str(), errno, strerror( errno ) );
      }

   fputc( bedFileMagic1, pFile );
   fputc( bedFileMagic2, pFile );
   fputc( layout_, pFile );
   int genotypeByte;
   int genotype;

   if ( layout_ == ColumnMajor )
      {
      for( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         for( size_t iIndividual=0; iIndividual<cIndividualsRead; )
            {
            genotypeByte = 0;
            for( int i=0; i<4; i++ )
               {
               if ( iIndividual >= cIndividualsRead )
                  {
                  break;
                  }
               genotype = BedGenotypeFromlimix::mfloat_t( rgBinaryGenotype[iIndividual][iSnp] );
               genotypeByte += (genotype << (i*2));
               ++iIndividual;
               }
            fputc( genotypeByte, pFile );
            }
         }
      }
   else
      {
      for( size_t iIndividual=0; iIndividual<cIndividualsRead; ++iIndividual )
         {
         for( size_t iSnp=0; iSnp<cSnps;  )
            {
            genotypeByte = 0;
            for( int i=0; i<4; i++ )
               {
               if ( iSnp >= cSnps )
                  {
                  break;
                  }
               genotype = BedGenotypeFromlimix::mfloat_t( rgBinaryGenotype[iIndividual][iSnp] );
               genotypeByte += (genotype << (i*2));
               ++iSnp;
               }
            fputc( genotypeByte, pFile );
            }
         }
      }

   fclose( pFile );
   }
#endif

void CPlinkFile::ReadTransposedFiles()
   {
   ReadTFamFile();
   ReadTPedFile();

   size_t cTPedIndividuals = rgTPed[0].rgSnps.size();
   size_t cTFamIndividuals = rgFam.size();
   if ( cTFamIndividuals != cTPedIndividuals )
      {
      Fatal( "Inconsistent individual counts in PLINK file set.  TFAM has %d individuals and TPED has %d individuals", cTFamIndividuals, cTPedIndividuals );
      }

   // ProduceFilteredData* routines need cIndividualsRead, cPhenotypes, cSnps set prior to entry
   //DEBUG:  Assert Phenotype info is already setup
   if ( (cPhenotypes != 1) || (ind_phenotype != 0) || (phenotypeLabels.size() != 1) )
      {
      Fatal( "Phenotype info not properly setup, expected 1, 0, 1 and found %d, %d, %d", cPhenotypes, ind_phenotype, phenotypeLabels.size() );
      }

   cIndividualsRead = cTFamIndividuals ;
   cSnpsRead = rgTPed.size();
   SelectIndividualsFromRgFam();
   SelectSnpsFromTransposedFiles();
   ProduceFilteredDataFromTransposedFiles();
   }

void CPlinkFile::CreateNaturalColumnMajorSnpArray()
   {
   // Create a 'Fortran' style array of floats for Snps
   //  we want the layout to be the same snp for all individuals are linear in memory.
   //        Individual1, Individual2, Individual3, ...
   //  snp1
   //  snp2
   //  snp3
   //    access to is by snpArray[ (iSnp*cPedIndividuals) + iIndividual ]
   //
   size_t cArrayElements = cIndividuals *cSnps;
   snpArray = new limix::mfloat_t[ cArrayElements ];

   for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
      {
      // Create the array of genotypes (limix::mfloat_ts) from SNPs
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      for ( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         size_t selectedSnp = selectedSnps[ iSnp ];
         limix::mfloat_t genotype = mfloat_tFromSnpNuculeotides( rgMap[ selectedSnp ].snpInfo.majorAllele, rgPed[ selectedIndividual ].rgSnps[ selectedSnp ] );
         size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnp );
         snpArray[ index ] = genotype;
         }
      }
   }

void CPlinkFile::CreateNaturalColumnMajorPhenArray()
   {
   // Create a 'Fortran' style array of floats for Phenotypes
   //  we want the layout to be the same snp for all individuals are linear in memory.
   //        Individual1, Individual2, Individual3, ...
   //  phen1
   //  phen2
   //  phen3
   //    access to is by phenArray[ (iPhenotype*cPedIndividuals) + iIndividual ]
   //
   size_t cArrayElements = cIndividuals * cPhenotypes;
   phenArray = new limix::mfloat_t[ cArrayElements ];

   for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
      {
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      for( size_t iPhenotype=0; iPhenotype<cPhenotypes; ++iPhenotype )
         {
         limix::mfloat_t phenotype;
         if ( alternatePhenotypeParameters.fUseAlternatePhenotype )
            {
            phenotype = alternatePhenotypeFile.PhenotypeValue( rgPed[ selectedIndividual ].idFamily, rgPed[ selectedIndividual ].idIndividual, alternatePhenotypeParameters.selectedPhenotypeIndex );
            }
         else
            {
            phenotype = rgPed[ selectedIndividual ].phenotype;
            }
         size_t index = ColumnMajorIndex( cIndividuals, cPhenotypes, iIndividual, iPhenotype );
         phenArray[ index ] = phenotype;
         }
      }
   }

#if 0
void CPlinkFile::CreateBinaryColumnMajorSnpArray()
   {
   // Create a 'Fortran' style array of floats for SNPs
   //  we want the layout to be the same snp for all individuals are linear in memory.
   //        Individual1, Individual2, Individual3, ...
   //  SNP1
   //  SNP2
   //  SNP3
   //    access to is by snpArray[ (iSnp*cIndividuals) + iIndividual ]
   //
   size_t cArrayElements = cIndividuals * cSnps;   
   snpArray = new limix::mfloat_t[ cArrayElements ];

   for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
      {
      // Create the array of genotypes (limix::mfloat_ts) from SNPs
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];

      for ( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         size_t selectedSnp = iSnp;
         if ( snpFilter.fUseSnpFilter )
            {
            selectedSnp = selectedSnps[ iSnp ];
            }

         limix::mfloat_t genotype = rgBinaryGenotype[selectedIndividual][selectedSnp];
         size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnp );
         snpArray[ index ] = genotype;
         }
      }
   }
#endif

#if 0
void CPlinkFile::CreateBinaryColumnMajorSnpArray3()
   {
   // Convert the STL std::vectors to linear memory for use by MKL
   //  we want the layout to be the same snp for all individuals are linear in memory.
   //        Individual1, Individual2, Individual3, ...
   //  SNP1
   //  SNP2
   //  SNP3
   //    access to is by snpArray[ (iSnp*cIndividuals) + iIndividual ]
   //    the offset is computed by ColumnMajorIndex(...)
   //
   if ( selectedSnps.size() != cSnps )
      {
      Fatal( "SNP selection not properly initialized.  Unable to proceed." );
      }

   size_t cArrayElements = cIndividuals * cSnps;
   try
      {
      snpArray = new limix::mfloat_t[ cArrayElements ];
      }
   catch( std::exception& e )
      {
#if defined( _MSC_VER )    // Windows/VC uses a %Iu specifier for size_t
      const char *szFmt1 = "  %Iu individuals * %Iu SNPs * %d sizeof(limix::mfloat_t) => %Iu";
#else                      // Linux/g++ uses a %zu specifier for size_t
      const char *szFmt1 = "  %zu individuals * %zu SNPs * %d sizeof(limix::mfloat_t) => %zu";
#endif
      Error( "Failed memory allocation for data array. exception => [%s]", e.what() );
      Error( szFmt1, cIndividuals, cSnps, sizeof( limix::mfloat_t ), cArrayElements * sizeof( limix::mfloat_t ) );
      Fatal( "Exit program" );
      }
   for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
      {
      // Create the array of genotypes (limix::mfloat_ts) from SNPs
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];

      for ( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         size_t selectedSnp = selectedSnps[ iSnp ];

         limix::mfloat_t genotype = rgBinaryGenotype[selectedIndividual][selectedSnp];
         size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnp );
         snpArray[ index ] = genotype;
         }
      }
   }
#endif

void CPlinkFile::CreateBinaryColumnMajorPhenArray()
   {
   //  The PLINK binary data for Phenotype is the same as the Transposed data
   CreateTransposedColumnMajorPhenArray();
   }

bool CPlinkFile::FSnpHasVariationTransposedFile( size_t iSnp )
   {
   SnpInfo& snpInfo = rgTPed[iSnp].snpInfo;
   char majorAllele = snpInfo.majorAllele;
   int hasHomozygousMajor = 0;
   int hasHomozygousMinor = 0;
   int hasHeterozygous = 0;

   std::vector< SnpNucleotides >& snpNucs = rgTPed[iSnp].rgSnps;

   for ( size_t iIndividual=0; iIndividual<selectedIndividuals.size(); ++iIndividual )
      {
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      Genotype genotype = GenotypeFromSnpNucleotides( majorAllele, snpNucs[ selectedIndividual ] );
      switch( genotype )
         {
      case missingGenotype:
         break;
      case homozygousMajor:
         ++hasHomozygousMajor;
         if ( hasHomozygousMinor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case homozygousMinor:
         ++hasHomozygousMinor;
         if ( hasHomozygousMajor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case heterozygous:
         ++hasHeterozygous;
         if ( hasHomozygousMajor || hasHomozygousMinor )
            {
            return( true );
            }
         break;
      default:
         Fatal( "Bad genotype for Individual/Snp [%s %s/%s]", 
                  rgFam[ selectedIndividual ].idFamily.c_str(), 
                  rgFam[ selectedIndividual ].idIndividual.c_str(),
                  snpInfo.idSnp.c_str() );
         break;
         }
      }
   return( false );
   }

void CPlinkFile::SelectSnpsFromTransposedFiles()
   {
   // Select the SNPs to propagate based on filtering
   //   and the individuals that are in the study sample
   //   If there is no genotype variation within the SNP
   //   in the study sample, remove the SNP from the study
   switch( snpFilter.filterType )
      {
   case SnpFilterOptions::FilterByNone:
      // No SNP filter applied.  Filter _ONLY_ 'constant SNPs'
      selectedSnps.reserve( cSnpsRead );
      for ( size_t iSnp=0; iSnp<cSnpsRead; ++iSnp )
         {
         if ( FSnpHasVariationTransposedFile( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            SnpInfo& snpInfo = rgTPed[iSnp].snpInfo;
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", snpInfo.idSnp.c_str() );
            }
         }
      break;
   case SnpFilterOptions::FilterByJob:
      {
      // No extract file, so it must be a sequential subset 
      //   created from 'Job' and JobNumber
      size_t firstSnp = snpFilter.IndexToFirstSnpInJob( cSnpsRead );
      size_t lastSnp = snpFilter.IndexToLastSnpInJob( cSnpsRead );
      selectedSnps.reserve( lastSnp - firstSnp );

      for( size_t iSnp=firstSnp; iSnp<lastSnp; ++iSnp )
         {
         if ( FSnpHasVariationTransposedFile( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            SnpInfo& snpInfo = rgTPed[iSnp].snpInfo;
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", snpInfo.idSnp.c_str() );
            }
         }
      }
      break;
   case SnpFilterOptions::FilterByFilelist:
   case SnpFilterOptions::FilterByTopN:
      // match the snps in snpFilter.snpsToProcess against the snps
      //   we have the map of SnpIds to index in indexOfSnpIds
      //   we have the list of SnpIds we want in snpFilter.snpIdsToExtract
      //   get the indexes put into selectedSnps (and then sort?) to be in original order
      selectedSnps.reserve( snpFilter.snpIdsToExtract.size() );
      for ( size_t i=0; i<snpFilter.snpIdsToExtract.size(); ++i )
         {
         if ( indexOfSnpIds.count( snpFilter.snpIdsToExtract[i] ) == 0 )
            {
            Warn( "SNP not found.  Cannot extract SNP information for [%s].  Skipping...", snpFilter.snpIdsToExtract[i].c_str() );
            }
         else
            {
            size_t iSnp = indexOfSnpIds[ snpFilter.snpIdsToExtract[i] ];
            if ( FSnpHasVariationTransposedFile( iSnp ) )
               {
               selectedSnps.push_back( iSnp );
               }
            else
               {
               SnpInfo& snpInfo = rgTPed[iSnp].snpInfo;
               Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", snpInfo.idSnp.c_str() );
               }
            }
         }
      sort( selectedSnps.begin(), selectedSnps.end() );
      break;
   default:
      Fatal( "Unexpected snpFilter type.  %d", snpFilter.filterType );
      }

   cSnps = selectedSnps.size();
   if ( cSnps == 0 )
      {
      Fatal( "No SNPs passed filter criteria." );
      }
   }

void CPlinkFile::CreateTransposedColumnMajorSnpArray()
   {
   // Create a 'Fortran' style (Column Major) array of floats for Snps
   //  we want the layout to be the same snp for all individuals are linear in memory.
   //        Individual1, Individual2, Individual3, ...
   //  snp1    addr0,           addr1,      addr2,  ...
   //  snp2   i*cIndividuals,  +1,          +2,
   //  snp3
   //    access to is by snpArray[ ColumnMajor( cIndividuals, cSnps, iIndividual, iSnp ) ]
   //       which equals snpArray[ (iSnp*cIndividuals) + iIndividual ]
   //
   size_t cArrayElements = cIndividuals * cSnps;
   snpArray = new limix::mfloat_t[cArrayElements];

   for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
      {
      // Create the array of genotypes (limix::mfloat_ts) from SNPs
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      for ( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         size_t selectedSnp = selectedSnps[ iSnp ];
         limix::mfloat_t genotype = mfloat_tFromSnpNuculeotides( rgTPed[ selectedSnp ].snpInfo.majorAllele, rgTPed[selectedSnp].rgSnps[ selectedIndividual ] );
         size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnp );
         snpArray[ index ] = genotype;
         }
      }
   }

void CPlinkFile::CreateTransposedColumnMajorPhenArray()
   {
   // Create a 'Fortran' style array of floats for Phenotypes
   //  we want the layout to be the same snp for all individuals are linear in memory.
   //        Individual1, Individual2, Individual3, ...
   //  phen1
   //  phen2
   //  phen3
   //    access to is by phenArray[ (iPhenotype*cPedIndividuals) + iIndividual ]
   //
   size_t cArrayElements = cIndividuals * cPhenotypes;   
   phenArray = new limix::mfloat_t[ cArrayElements ];

   for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
      {
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      for( size_t iPhenotype=0; iPhenotype<cPhenotypes; ++iPhenotype )
         {
         limix::mfloat_t phenotype;
         if ( alternatePhenotypeParameters.fUseAlternatePhenotype )
            {
            phenotype = alternatePhenotypeFile.PhenotypeValue( rgFam[ selectedIndividual ].idFamily, rgFam[ selectedIndividual ].idIndividual, alternatePhenotypeParameters.selectedPhenotypeIndex );
            }
         else
            {
            phenotype = rgFam[ selectedIndividual ].phenotype;
            }
         size_t index = ColumnMajorIndex( cIndividuals, cPhenotypes, iIndividual, iPhenotype );
         phenArray[ index ] = phenotype;
         }
      }
   }

void CPlinkFile::CreateDosageColumnMajorPhenArray()
   {
   // Phenotype is in the AlternatePhenotype file or a FAM file same as the Transposed.
   CreateTransposedColumnMajorPhenArray();   
   }

void CPlinkFile::CreateDosageColumnMajorSnpArray( CPlinkDatFile& datFile )
   {
   // Create a Column Major array of floats for Dosage values (genotypes)
   size_t cArrayElements = cIndividuals * cSnps;
   snpArray = new limix::mfloat_t[cArrayElements];

   for ( size_t iIndividual=0; iIndividual<cIndividuals; ++iIndividual )
      {
      // Create the array of genotypes (limix::mfloat_ts) from SNPs
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      for ( size_t iSnp=0; iSnp<cSnps; ++iSnp )
         {
         size_t selectedSnp = selectedSnps[ iSnp ];
         limix::mfloat_t genotype = mfloat_tFromSnpProbabilities( datFile.rgDat[ selectedSnp ].rgSnpProbabilities[ selectedIndividual ] );
         size_t index = ColumnMajorIndex( cIndividuals, cSnps, iIndividual, iSnp );
         snpArray[ index ] = genotype;
         }
      }
   }

limix::mfloat_t CPlinkFile::mfloat_tFromSnpNuculeotides( char& majorAllele_, SnpNucleotides& snp )
   {
   // We _KNOW_ the snp is well formed!  (
   if ( (majorAllele_ == '0') || (snp.alleles[0] == '0') )
      {
      return ( unknownOrMissing );
      }
   char chMajor = toupper( majorAllele_ );
   char ch1 = toupper( snp.alleles[0] );
   char ch2 = toupper( snp.alleles[1] );
   if ( ch1 != ch2 )
      {
      return( heterozygousAllele );
      }
   if ( ch1 == '0' )
      {
      return( unknownOrMissing );
      }
   else if ( ch1 == chMajor )
      {
      return( homozygousPrimaryAllele );
      }
   return( homozygousSecondaryAllele );
   }

limix::mfloat_t CPlinkFile::mfloat_tFromSnpProbabilities( SnpProbabilities& snpP )
   {
   limix::mfloat_t rc;
   if ( (snpP.probabilityOfHomozygousMinor == -9.0) || (snpP.probabilityOfHeterozygous == -9.0) )
      {
      rc = unknownOrMissing;
      }
   else
      {
      rc = (snpP.probabilityOfHomozygousMinor * 2) + snpP.probabilityOfHeterozygous;
      }
   return( rc );
   }

void CPlinkFile::ComputeNaturalSnpAlleleChars()
   {
   std::map< char, size_t > alleleMap;

   /*
    *  Iterate along each snp and compute the allele frequency
    *    the most frequent snp is used to define the pre-dominate
    *    snp allele for this position.
    *    This is used when converting a genotype to a floating point
    *    value.
    */
   size_t cPedIndividuals = rgPed.size();
   size_t cPedSnps = rgMap.size();

   for( size_t iSnp=0; iSnp<cPedSnps; ++iSnp )
      {
      alleleMap.clear();
      for ( size_t iIndividual=0; iIndividual<cPedIndividuals; ++iIndividual )
         {
         char ch0 = rgPed[iIndividual].rgSnps[iSnp].alleles[0];
         char ch1 = rgPed[iIndividual].rgSnps[iSnp].alleles[1];
         if ( ch0 != '0' )
            {
            // only need to test one allele since we tested for missing consistency during read
            alleleMap[ ch0 ] += 1;
            alleleMap[ ch1 ] += 1;
            }
         }
      /*
       * Now we have all the alleles used and a count of each.
       */
      if ( alleleMap.size() < 1 ) 
         {
         Fatal( "Found no allele variants for Snp [%s]", rgMap[iSnp].snpInfo.idSnp.c_str() );
         }
      if ( alleleMap.size() > 2 )
         {
         Fatal( "Found too many allele variants for Snp [%s]", rgMap[iSnp].snpInfo.idSnp.c_str() );
         }

      /*
       *  So get the predominant allele and it becomes majorAllele in rgMap[iSnp].snpAlleleChars
       *  there are at most two in the map so get them.
       */
      //auto pos = alleleMap.begin();
      map<char, size_t>::iterator pos = alleleMap.begin();
      rgMap[iSnp].snpInfo.majorAllele = pos->first;
      size_t countOfAllele0 = pos->second;
      if ( ++pos != alleleMap.end() )
         {
         if ( pos->second > countOfAllele0 )
            {
            rgMap[iSnp].snpInfo.minorAllele = rgMap[iSnp].snpInfo.majorAllele;
            rgMap[iSnp].snpInfo.majorAllele = pos->first;
            }
         else
            {
            rgMap[iSnp].snpInfo.minorAllele = pos->first;
            }
         }
      }
   }

void CPlinkFile::ReadMapFile()
   {
   std::string filename;
   if ( this->fileType == Natural )
      {
      filename = baseFilename + ".map";
      }
   else if ( this->fileType == Binary )
      {
      filename = baseFilename + ".bim";
      }

   Verbose( "              Processing PLink file: [%s]", filename.c_str() );
   CPlinkLexer lex( filename );
   CToken tok;
   MapRecord maprecord;

   lex.NextToken( tok );
   while ( tok.type != tokEOF )        // for each line until EOF
      {
      if ( tok.type == tokEOL )        // blank line...or comment line or ???
         {
         lex.NextToken( tok );
         continue;
         }

      lex.ExpectChromosome( tok, maprecord.snpInfo.idChromosome, maprecord.snpInfo.iChromosome );
      lex.ExpectId( tok, maprecord.snpInfo.idSnp, "Snp ID" );
      lex.Expectlimix::mfloat_t( tok, maprecord.snpInfo.geneticDistance, "Genetic Distance" );
      lex.ExpectInt( tok, maprecord.snpInfo.basepairPosition, "Basepair Position" );
      if ( this->fileType == Natural )
         {
         if ( tok.type != tokEOL )
            {
            Fatal( "Expecting <EOL> after Basepair Position at line %d:%d.  Found [%s]", tok.line, tok.column, tok.text.c_str() );
            }
         
         lex.NextToken( tok );      // consume the <EOL>
         maprecord.snpInfo.majorAllele = '\0';
         maprecord.snpInfo.minorAllele = '\0';
         }
      else
         {
         lex.ExpectSnpAlleles( tok, maprecord.snpInfo.majorAllele, maprecord.snpInfo.minorAllele );
         if ( tok.type != tokEOL )
            {
            Fatal( "Expecting <EOL> after SNP alleles at line %d:%d.  Found [%s]", tok.line, tok.column, tok.text.c_str() );
            }

         lex.NextToken( tok );      // consume the <EOL>
         }

      AddSnpIdToIndex( maprecord.snpInfo.idSnp, rgMap.size() );
      rgMap.push_back( maprecord );
      }

   if ( fCreateReadValidationFiles )         // test global to write validation file 
      {
      std::string s = filename + ".validate.txt";
      WriteMapFile( s );
      }
   }

void CPlinkFile::WriteMapFile( const std::string& mapFileName )
   {
   FILE *pFile;

   if ( mapFileName.empty() )
      {
      Fatal( "No output filename to open " );
      }

   pFile = fopen( mapFileName.c_str(), "wt" );          // write ascii

   if ( !pFile )
      {
      std::string fullPath = FullPath( mapFileName );
      Fatal( "Cannot open output file [%s].  \n  CRT Error %d: %s", fullPath.c_str(), errno, strerror( errno ) );
      }

   size_t cMapRows = rgMap.size();
   for ( size_t i=0; i<cMapRows; ++i )
      {
      MapRecord *pmr = &rgMap[i];
      fprintf( pFile, "%-2.2s %s %g %d", pmr->snpInfo.idChromosome.c_str(), pmr->snpInfo.idSnp.c_str(), pmr->snpInfo.geneticDistance, pmr->snpInfo.basepairPosition );
      if ( this->fileType == Binary )
         {
         fprintf( pFile, " %c %c", pmr->snpInfo.minorAllele, pmr->snpInfo.majorAllele );
         }
      fprintf( pFile, "\n" );
      }
   fclose( pFile );
   }

void CPlinkFile::ReadPedFile()
   {
   std::string filename = baseFilename + ".ped";
   Verbose( "              Processing PLink file: [%s]", filename.c_str() );
   CPlinkLexer lex( filename );
   CToken tok;
   PedRecord pedrecord;
   size_t cSnps = 0;

   lex.NextToken( tok );
   while ( tok.type != tokEOF )        // for each line until EOF
      {
      if ( tok.type == tokEOL )        // blank line...or comment line or ???
         {
         lex.NextToken( tok );
         continue;
         }
      lex.ExpectId( tok, pedrecord.idFamily, "FamilyID" );
      lex.ExpectId( tok, pedrecord.idIndividual, "IndividualID" );
      lex.ExpectId( tok, pedrecord.idPaternal, "PaternalID" );
      lex.ExpectId( tok, pedrecord.idMaternal, "MaternalID" );
      lex.ExpectSex( tok, pedrecord.sex );
      lex.ExpectPhenotype( tok, pedrecord.phenotype, phenotypeValueType );

      // Now get the genotype array
      pedrecord.rgSnps.clear();
      SnpNucleotides s;

      while( tok.type != tokEOF )
         {
         if ( tok.type == tokEOL )
            {
            lex.NextToken( tok );
            break;
            }
         lex.ExpectSnpNucleotides( tok, s.alleles[0], s.alleles[1] );
         pedrecord.rgSnps.push_back( s );
         }

      if ( pedrecord.rgSnps.empty() )
         {
         Fatal( "Expected genotype information in line %d.  Found none.", tok.line );
         }
      if ( cSnps == 0 )
         {
         cSnps = pedrecord.rgSnps.size();
         }
      else if ( pedrecord.rgSnps.size() != cSnps )
         {
         Fatal( "Expected the same number of genotypes in each row.  Found %d in row 1 and %d in row %d",
            cSnps, pedrecord.rgSnps.size(), tok.line );
         }

      AddToFamilyIdAndIndividualIdIndex( pedrecord.idFamily, pedrecord.idIndividual, rgPed.size() );
      rgPed.push_back( pedrecord );
      }

   if ( fCreateReadValidationFiles )         // test global to write validation file 
      {
      std::string s = filename + ".validate.txt";
      WritePedFile( s );
      }
   }

char *SzFromPhenotype( limix::mfloat_t phenotype )
   {
   static char szPhenotypeArray[4][32];
   static int iPhenotypeArray = 0;

   char *pch = szPhenotypeArray[ iPhenotypeArray ];
   iPhenotypeArray = (iPhenotypeArray+1) & 3;

   if ( phenotype != phenotype )
      {     // it's a NaN so we should emit the 'missingPhenotypeValue'
      phenotype = missingPhenotypeValue;
      }
   sprintf( pch, "%3g", phenotype );
   return( pch );
   }

void CPlinkFile::WritePedFile( const std::string& pedFile )
   {
   FILE *pFile;

   if ( pedFile.empty() )
      {
      Fatal( "No output filename to open " );
      }

   pFile = fopen( pedFile.c_str(), "wt" );          // write ascii

   if ( !pFile )
      {
      std::string fullPath = FullPath( pedFile );
      Fatal( "Cannot open output file [%s].  \n  CRT Error %d: %s", fullPath.c_str(), errno, strerror( errno ) );
      }

   size_t cPedIndividuals = rgPed.size();
   size_t cPedSnps = rgMap.size();
   for ( size_t iIndividual=0; iIndividual<cPedIndividuals; ++iIndividual )
      {
      PedRecord *pr = &rgPed[iIndividual];
      fprintf( pFile, "%s %s %s %s %1d",  pr->idFamily.c_str(), pr->idIndividual.c_str(), pr->idPaternal.c_str(), pr->idMaternal.c_str(), pr->sex );
      fprintf( pFile, " %s", SzFromPhenotype( pr->phenotype ) ); 
      for( size_t iSnp=0; iSnp<pr->rgSnps.size(); ++iSnp )
         {
         fprintf( pFile, "  %c %c", pr->rgSnps[iSnp].alleles[0], pr->rgSnps[iSnp].alleles[1] );
         }
      fprintf( pFile, "\n" );
      }

   fclose( pFile );
   }

void CPlinkFile::ReadFamFile()
   {
   std::string filename = baseFilename + ".fam";
   ReadFamFile( filename );
   }

void CPlinkFile::ReadFamFile( const std::string& filename )
   {
   size_t cIndividuals = 0;

   // first 6 columns are the same as the PED file, without the genotype information
   Verbose( "              Processing PLink file: [%s]", filename.c_str() );
   CPlinkLexer lex( filename );
   CToken tok;
   FamRecord famRecord;

   lex.NextToken( tok );
   while ( tok.type != tokEOF )        // for each line until EOF
      {
      if ( tok.type == tokEOL )        // blank line...or comment line or ???
         {
         lex.NextToken( tok );
         continue;
         }
      lex.ExpectId( tok, famRecord.idFamily, "FamilyID" );
      lex.ExpectId( tok, famRecord.idIndividual, "IndividualID" );
      lex.ExpectId( tok, famRecord.idPaternal, "PaternalID" );
      lex.ExpectId( tok, famRecord.idMaternal, "MaternalID" );
      lex.ExpectSex( tok, famRecord.sex );
      lex.ExpectPhenotype( tok, famRecord.phenotype, phenotypeValueType );
      
      AddToFamilyIdAndIndividualIdIndex( famRecord.idFamily, famRecord.idIndividual, cIndividuals );
      rgFam.push_back( famRecord );
      ++cIndividuals;
      }

   if ( fCreateReadValidationFiles )         // test global to write validation file 
      {
      std::string s = filename + ".validate.txt";
      WriteFamFile( s );
      }
   }

void CPlinkFile::WriteFamFile( const std::string& famFile )
   {
   FILE *pFile;

   if ( famFile.empty() )
      {
      Fatal( "No output filename to open " );
      }

   pFile = fopen( famFile.c_str(), "wt" );         // write ascii

   if ( !pFile )
      {
      std::string fullPath = FullPath( famFile );
      Fatal( "Cannot open output file [%s].  \n  CRT Error %d: %s", fullPath.c_str(), errno, strerror( errno ) );
      }

   size_t individualMax = rgFam.size();
   for ( size_t iIndividual=0; iIndividual<individualMax; ++iIndividual )
      {
      FamRecord *pr = &rgFam[iIndividual];
      fprintf( pFile, "%s %s %s %s %1d",  pr->idFamily.c_str(), pr->idIndividual.c_str(), pr->idPaternal.c_str(), pr->idMaternal.c_str(), pr->sex );
      fprintf( pFile, " %s", SzFromPhenotype( pr->phenotype ) );
      fprintf( pFile, "\n" );
      }

   fclose( pFile );
   }

void CPlinkFile::ReadTPedFile()
   {
   // First 4 columns are the same as the MAP file + the SNP data across the individual base
   std::string filename = baseFilename + ".tped";
   Verbose( "              Processing PLink file: [%s]", filename.c_str() );
   CPlinkLexer lex( filename );
   CToken tok;
   TPedRecord tpedRecord;
   map< char, size_t > alleleMap;
   size_t cSnps = 0;

   lex.NextToken( tok );
   while ( tok.type != tokEOF )        // for each line until EOF
      {
      if ( tok.type == tokEOL )        // blank line...or comment line or ???
         {
         lex.NextToken( tok );
         continue;
         }

      lex.ExpectChromosome( tok, tpedRecord.snpInfo.idChromosome, tpedRecord.snpInfo.iChromosome );
      lex.ExpectId( tok, tpedRecord.snpInfo.idSnp, "Snp ID" );
      lex.Expectlimix::mfloat_t( tok, tpedRecord.snpInfo.geneticDistance, "Genetic Distance" );
      lex.ExpectInt( tok, tpedRecord.snpInfo.basepairPosition, "Basepair Position" );

      //
      // Now get the SNPs array
      tpedRecord.rgSnps.clear();
      alleleMap.clear();
      SnpNucleotides s;

      while( tok.type != tokEOF )
         {
         if ( tok.type == tokEOL )
            {
            lex.NextToken( tok );
            break;
            }
         lex.ExpectSnpNucleotides( tok, s.alleles[0], s.alleles[1] );

         // We have the validated snp information in s.  
         tpedRecord.rgSnps.push_back( s );

         // Track the information to compute the predominant allele
         if ( s.alleles[0] != '0' )
            {
            // We only need to test one allele since we validated that if one is missing, both are missing.
            alleleMap[ s.alleles[0] ] += 1;
            alleleMap[ s.alleles[1] ] += 1;
            }
         }

      if ( tpedRecord.rgSnps.empty() )
         {
         Fatal( "Expected genotype information in line %d.  Found none.", tok.line );
         }
      if ( cSnps == 0 )
         {
         cSnps = tpedRecord.rgSnps.size();
         }
      else if ( tpedRecord.rgSnps.size() != cSnps )
         {
         Fatal( "Expected the same number of genotypes in each row.  Found %d in row 1 and %d in row %d",
            cSnps, tpedRecord.rgSnps.size(), tok.line );
         }

      /*
       * Use the Allele map to compute the predominate information
       */
      if ( alleleMap.size() < 1 )
         {
         Fatal( "Found no allele variants for Snp [%s]", tpedRecord.snpInfo.idSnp.c_str() );
         }
      if ( alleleMap.size() > 2 )
         {
         Fatal( "Found too many allele variants for Snp[%s]", tpedRecord.snpInfo.idSnp.c_str() );
         }

      //auto pos = alleleMap.begin();     // c++ 0x syntax
      map<char, size_t>::iterator pos = alleleMap.begin();
      tpedRecord.snpInfo.majorAllele = pos->first;
      size_t countOfAllele0 = pos->second;
      if ( ++pos != alleleMap.end() )
         {
         if ( pos->second > countOfAllele0 )
            {
            tpedRecord.snpInfo.minorAllele = tpedRecord.snpInfo.majorAllele;
            tpedRecord.snpInfo.majorAllele = pos->first;
            }
         else
            {
            tpedRecord.snpInfo.minorAllele = pos->first;
            }
         }

      AddSnpIdToIndex( tpedRecord.snpInfo.idSnp, rgTPed.size() );
      rgTPed.push_back( tpedRecord );
      }

   if ( fCreateReadValidationFiles )         // test global to write validation file 
      {
      std::string s = filename + ".validate.txt";
      WriteTPedFile( s );
      }
   }

void CPlinkFile::WriteTPedFile( const std::string& tpedFileName )
   {
   FILE *pFile;

   if ( tpedFileName.empty() )
      {
      Fatal( "No output filename to open " );
      }

   pFile = fopen( tpedFileName.c_str(), "wt" );          // write ascii

   if ( !pFile )
      {
      std::string fullPath = FullPath( tpedFileName );
      Fatal( "Cannot open output file [%s].  \n  CRT Error %d: %s", fullPath.c_str(), errno, strerror( errno ) );
      }

   for ( size_t i=0; i<rgTPed.size(); ++i )
      {
      TPedRecord *ptr = &rgTPed[i];
      fprintf( pFile, "%-2.2s %s %g %d", ptr->snpInfo.idChromosome.c_str(), ptr->snpInfo.idSnp.c_str(), ptr->snpInfo.geneticDistance, ptr->snpInfo.basepairPosition );
      for ( size_t iSnp=0; iSnp<ptr->rgSnps.size(); ++iSnp )
         {
         fprintf( pFile, "  %c %c", ptr->rgSnps[iSnp].alleles[0], ptr->rgSnps[iSnp].alleles[1] );
         }
      fprintf( pFile, "\n" );
      }
   fclose( pFile );
   }

void CPlinkFile::ReadTFamFile()
   {
   // TFAM is identical to FAM file
   std::string filename = baseFilename + ".tfam";
   ReadFamFile( filename );
   }

bool CPlinkFile::FSnpHasVariation( size_t iSnp )
   {
   char majorAllele = rgMap[ iSnp ].snpInfo.majorAllele;
   int hasHomozygousMajor = 0;
   int hasHomozygousMinor = 0;
   int hasHeterozygous = 0;

   for ( size_t iIndividual=0; iIndividual<selectedIndividuals.size(); ++iIndividual )
      {
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      Genotype genotype = GenotypeFromSnpNucleotides( majorAllele, rgPed[ selectedIndividual ].rgSnps[ iSnp ] );
      switch( genotype )
         {
      case missingGenotype:
         break;
      case homozygousMajor:
         ++hasHomozygousMajor;
         if ( hasHomozygousMinor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case homozygousMinor:
         ++hasHomozygousMinor;
         if ( hasHomozygousMajor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case heterozygous:
         ++hasHeterozygous;
         if ( hasHomozygousMajor || hasHomozygousMinor )
            {
            return( true );
            }
         break;
      default:
         Fatal( "Bad genotype for Individual/Snp [%s %s/%s]", 
                  rgPed[ selectedIndividual ].idFamily.c_str(), 
                  rgPed[ selectedIndividual ].idIndividual.c_str(),
                  rgMap[ iSnp ].snpInfo.idSnp.c_str() );
         break;
         }
      }
   return( false );
   }

#if 0
bool CPlinkFile::FSnpHasVariationBinaryFile( size_t iSnp )
   {
   SnpInfo& snpInfo = rgMap[ iSnp ].snpInfo;
   int hasHomozygousMajor = 0;
   int hasHomozygousMinor = 0;
   int hasHeterozygous = 0;

   for ( size_t iIndividual=0; iIndividual<selectedIndividuals.size(); ++iIndividual )
      {
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      limix::mfloat_t limix::mfloat_tGenotype = rgBinaryGenotype[selectedIndividual][iSnp];

      if ( limix::mfloat_tGenotype == homozygousPrimaryAllele )
         {
         ++hasHomozygousMajor;
         if ( hasHomozygousMinor || hasHeterozygous )
            {
            return( true );
            }
         }
      else if ( limix::mfloat_tGenotype == heterozygousAllele )
         {
         ++hasHeterozygous;
         if ( hasHomozygousMajor || hasHomozygousMinor )
            {
            return( true );
            }
         }
      else if ( limix::mfloat_tGenotype == homozygousSecondaryAllele )
         {
         ++hasHomozygousMinor;
         if ( hasHomozygousMajor || hasHeterozygous )
            {
            return( true );
            }
         }
      }
      
   return( false );
   }
#endif

#if 0
bool CPlinkFile::FSnpHasVariation3( SnpInfo& snpInfo, std::vector< BedGenotype >& rgBedGenotypes )
   {
   /*
    * Review each SNP group of the BED file to identify SNP vs. SNC (constant)
    */

   int hasHomozygousMajor = 0;
   int hasHomozygousMinor = 0;
   int hasHeterozygous = 0;

   for ( size_t iIndividual=0; iIndividual<selectedIndividuals.size(); ++iIndividual )
      {
      size_t selectedIndividual = selectedIndividuals[ iIndividual ];
      BedGenotype bedGenotype = rgBedGenotypes[ selectedIndividual ];
      switch( bedGenotype )
         {
      case bedMissingGenotype:
         break;
      case bedHomozygousMajor:
         ++hasHomozygousMajor;
         if ( hasHomozygousMinor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case bedHomozygousMinor:
         ++hasHomozygousMinor;
         if ( hasHomozygousMajor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case bedHeterozygous:
         ++hasHeterozygous;
         if ( hasHomozygousMajor || hasHomozygousMinor )
            {
            return( true );
            }
         break;
      default:
         Fatal( "Bad genotype for Individual/Snp [%s %s/%s]", 
                  rgPed[ selectedIndividual ].idFamily.c_str(), 
                  rgPed[ selectedIndividual ].idIndividual.c_str(),
                  snpInfo.idSnp.c_str() );
         break;
         }
      }
   return( false );
   }
#endif

bool CPlinkFile::FSnpHasVariation4( SnpInfo& snpInfo, std::vector< BedGenotype >& rgBedGenotypes, std::vector< size_t >& rgSelected )
   {
   /*
    * Review each SNP group of the BED file to identify SNP vs. SNC (constant)
    */
   int hasHomozygousMajor = 0;
   int hasHomozygousMinor = 0;
   int hasHeterozygous = 0;

   for ( size_t iIndividual=0; iIndividual<rgSelected.size(); ++iIndividual )
      {
      size_t selectedIndividual = rgSelected[ iIndividual ];
      BedGenotype bedGenotype = rgBedGenotypes[ selectedIndividual ];
      switch( bedGenotype )
         {
      case bedMissingGenotype:
         break;
      case bedHomozygousMajor:
         ++hasHomozygousMajor;
         if ( hasHomozygousMinor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case bedHomozygousMinor:
         ++hasHomozygousMinor;
         if ( hasHomozygousMajor || hasHeterozygous )
            {
            return( true );
            }
         break;
      case bedHeterozygous:
         ++hasHeterozygous;
         if ( hasHomozygousMajor || hasHomozygousMinor )
            {
            return( true );
            }
         break;
      default:
         Fatal( "Bad genotype for Individual/Snp [%s %s/%s]", 
                  rgPed[ selectedIndividual ].idFamily.c_str(), 
                  rgPed[ selectedIndividual ].idIndividual.c_str(),
                  snpInfo.idSnp.c_str() );
         break;
         }
      }
   return( false );
   }

bool CPlinkFile::FSnpHasVariation4b( SnpInfo& snpInfo, limix::mfloat_t* pGenotype )
   {
   limix::mfloat_t previousGenotype = -1.0;

   for ( size_t iIndividual = 0; iIndividual < cIndividuals; ++iIndividual )
      {
      limix::mfloat_t genotype = *pGenotype;
      if ( genotype == genotype )
         {
         if ( previousGenotype == -1.0 )
            {
            previousGenotype = genotype;
            }
         else
            {
            if ( previousGenotype != genotype )
               {
               return( true );
               }
            }
         }
      ++pGenotype;;
      }
   return( false );
   }

void CPlinkFile::SelectIndividualsFromRgPed()
   {
   if ( !alternatePhenotypeParameters.fUseAlternatePhenotype )
      {
      Fatal( "Alternate Phenotype File Required!" );
      }

   /*
    *  Use the alternate phenotype file as the master individual list
    *    and lookup the individual genotype in the rgPed information
    *    and get the phenotype for this individual
    */
   size_t cAlternatePhenotypeIndividuals = alternatePhenotypeFile.CountOfIndividuals();
   individualLabels.reserve( cAlternatePhenotypeIndividuals );
   selectedIndividuals.reserve( cAlternatePhenotypeIndividuals );
   for ( size_t iIndividual=0; iIndividual < cAlternatePhenotypeIndividuals; ++iIndividual )
      {
      std::string key;
      alternatePhenotypeFile.KeyFromIndividualIndex( iIndividual, key );

      // verify there is phenotype associated with this record.
      limix::mfloat_t phenotype = alternatePhenotypeFile.PhenotypeValue( key );
      if ( phenotype != phenotype )
         {
         Warn( "Found 'missing' phenotype marker.  Excluding [%s] from the study", key.c_str() );
         continue;
         }

      // verify the individual exists in the PED file
      if ( indexOfFamilyIdAndIndividualId.count( key ) == 0 )
         {
         Warn( "Missing PED information.  Excluding [%s] from the study", key.c_str() );
         continue;
         }

      // verify the individual exists in the covariates file (if there is one)
      if ( this->covariatesFileData && !covariatesFileData->FKeyIndividualInDatFile( key ) )
         {
         Warn( "Individual from alternate phenotype file not found in Covariates file [%s]"
             "\nExcluding [%s] from the study", covariatesFileData->filename.c_str(), key.c_str() );
         continue;
         }

      size_t indexT = indexOfFamilyIdAndIndividualId[ key ];
      individualLabels.push_back( key );
      selectedIndividuals.push_back( indexT );
      }

   cIndividuals = selectedIndividuals.size();
   }

/*
 *  Inputs: alternatePhenotypeParameters
 *          
 * Outputs: sets std::vector<std::string> individualLabels
 *          sets map<std::string, size_t> selectedIndividuals - indicating which individuals from fam array are selected
 */
void CPlinkFile::SelectIndividualsFromRgFam()
   {
   if ( !alternatePhenotypeParameters.fUseAlternatePhenotype )
      {
      Fatal( "Alternate Phenotype File Required!" );
      }

   /*
    *  Use the alternate phenotype file as the master individual list
    *    and lookup the individual genotype in the rgFam information
    *    and get the phenotype for this individual
    */
   size_t cAlternatePhenotypeIndividuals = alternatePhenotypeFile.CountOfIndividuals();
   individualLabels.reserve( cAlternatePhenotypeIndividuals );
   selectedIndividuals.reserve( cAlternatePhenotypeIndividuals );
   for ( size_t iIndividual=0; iIndividual < cAlternatePhenotypeIndividuals; ++iIndividual )
      {
      std::string key;
      alternatePhenotypeFile.KeyFromIndividualIndex( iIndividual, key );

      // verify there is phenotype associated with this record.
      limix::mfloat_t phenotype = alternatePhenotypeFile.PhenotypeValue( key );
      if ( phenotype != phenotype )
         {
         Warn( "Found 'missing' phenotype marker.  Excluding [%s] from the study", key.c_str() );
         continue;
         }

      // verify the individual exists in the fam/tfam file
      if ( indexOfFamilyIdAndIndividualId.count( key ) == 0 )
         {
         Warn( "Individual from alternate phenotype file not found in .FAM or .TFAM file."
             "\nExcluding [%s] from the study", key.c_str() );
         continue;
         }

      // verify the individual exists in the covariates file (if there is one)
      if ( this->covariatesFileData && !covariatesFileData->FKeyIndividualInDatFile( key ) )
         {
         Warn( "Individual from alternate phenotype file not found in Covariates file [%s]"
             "\nExcluding [%s] from the study", covariatesFileData->filename.c_str(), key.c_str() );
         continue;
         }

      size_t indexT = indexOfFamilyIdAndIndividualId[ key ];
      individualLabels.push_back( key );
      selectedIndividuals.push_back( indexT );
      }

   cIndividuals = individualLabels.size();
   }

void CPlinkFile::SelectSnps( size_t cSnpsRead, SnpFilterOptions& snpFilter )
   {
   SelectSnps( cSnpsRead, snpFilter, this->selectedSnps );
#if 0
   if ( !snpFilter.fUseSnpFilter )
      {
      // No SNP filter applied.  Use all the SNPs
      if ( selectedSnps.size() > 0 )
         {
         Fatal( "Expected selectedSnps to be 0 at this point." );
         }

      selectedSnps.reserve( cSnpsRead );
      for( size_t iSnp = 0; iSnp<cSnpsRead; ++iSnp )
         {
         selectedSnps.push_back( iSnp );
         }
      cSnps = cSnpsRead;
      }
   else if ( snpFilter.extractFile.empty() )
      {
      // No extract file, so it must be a sequential subset 
      //   created from 'Job' and JobNumber
      snpFilter.sizeOfJob = ((int)cSnpsRead + snpFilter.numberOfJobs - 1) / snpFilter.numberOfJobs;    // round up for the size 
      size_t firstSnp = snpFilter.sizeOfJob * snpFilter.thisJobIndex;
      size_t lastSnp = min( (firstSnp + snpFilter.sizeOfJob), cSnpsRead );
      cSnps = lastSnp - firstSnp;
      selectedSnps.reserve( cSnps );

      for( size_t i=firstSnp; i<lastSnp; ++i )
         {
         selectedSnps.push_back(i);
         }
   
      if ( cSnps != selectedSnps.size() )
         {
         Fatal( "Error processing SelectSnps.  Expected %d SNPs, found %d SNPs", cSnps, selectedSnps.size() );
         }
      }
   else
      {
      // match the snps in snpFilter.snpsToProcess against the snps
      //   we have the map of SnpIds to index in indexOfSnpIds
      //   we have the list of SnpIds we want in snpFilter.snpsToProcess
      //   get the indexes put into selectedSnps (and then sort?)
      selectedSnps.reserve( snpFilter.snpsToProcess.size() );
      for ( size_t i=0; i<snpFilter.snpsToProcess.size(); ++i )
         {
         if ( indexOfSnpIds.count( snpFilter.snpsToProcess[i] ) == 0 )
            {
            Warn( "SNP not found.  Cannot extract SNP information for [%s].  Skipping...", snpFilter.snpsToProcess[i].c_str() );
            }
         else
            {
            size_t iSnp = indexOfSnpIds[ snpFilter.snpsToProcess[i] ];
            selectedSnps.push_back( iSnp );
            }
         }
      if ( selectedSnps.size() == 0 )
         {
         Fatal( "No SNPs matching filter -extract file contents found." );
         }

      sort( selectedSnps.begin(), selectedSnps.end() );
      cSnps = selectedSnps.size();
      }
#endif
   }

void CPlinkFile::SelectSnps( size_t cSnpsRead, SnpFilterOptions& snpFilter, std::vector< size_t >& snpsSelected )
   {
   // Select the SNPs to propagate based on filtering
   //   and the individuals that are in the study sample
   // NOTE:  This method DOES NOT filter on constant valued SNPs
   //        You must do that after this routine completes
   if ( snpsSelected.size() > 0 )
      {
      Fatal( "Expected snpsSelected to be 0 at this point." );
      }

   switch( snpFilter.filterType )
      {
   case SnpFilterOptions::FilterByNone:
      // No SNP filter applied.  Use all the SNPs
      snpsSelected.reserve( cSnpsRead );
      for( size_t iSnp = 0; iSnp<cSnpsRead; ++iSnp )
         {
         snpsSelected.push_back( iSnp );
         }
      break;
   case SnpFilterOptions::FilterByJob:
      {
      // No extract file, so it must be a sequential subset 
      //   created from 'Job' and JobNumber
      size_t firstSnp = snpFilter.IndexToFirstSnpInJob( cSnpsRead );
      size_t lastSnp = snpFilter.IndexToLastSnpInJob( cSnpsRead );
      selectedSnps.reserve( lastSnp - firstSnp );

      for( size_t iSnp=firstSnp; iSnp<lastSnp; ++iSnp )
         {
         snpsSelected.push_back( iSnp );
         }
      }
      break;
   case SnpFilterOptions::FilterByFilelist:
   case SnpFilterOptions::FilterByTopN:
      // match the snps in snpFilter.snpsToProcess against the snps
      //   we have the map of SnpIds to index in indexOfSnpIds
      //   we have the list of SnpIds we want in snpFilter.snpIdsToExtract
      //   get the indexes put into selectedSnps (and then sort?) to be in original order
      selectedSnps.reserve( snpFilter.snpIdsToExtract.size() );
      for ( size_t i=0; i<snpFilter.snpIdsToExtract.size(); ++i )
         {
         if ( indexOfSnpIds.count( snpFilter.snpIdsToExtract[i] ) == 0 )
            {
            Warn( "SNP not found.  Cannot extract SNP information for [%s].  Skipping...", snpFilter.snpIdsToExtract[i].c_str() );
            }
         else
            {
            size_t iSnp = indexOfSnpIds[ snpFilter.snpIdsToExtract[i] ];
            snpsSelected.push_back( iSnp );
            }
         }
      sort( snpsSelected.begin(), snpsSelected.end() );
      break;
   default:
      Fatal( "Unexpected snpFilter type.  %d", snpFilter.filterType );
      }

   cSnps = snpsSelected.size();
   if ( snpsSelected.size() == 0 )
      {
      Fatal( "No SNPs passed filter criteria." );
      }
   }

void CPlinkFile::ExtractSnpsFromRgTPed()
   {
   if ( selectedSnps.size() == 0 )
      {
      Fatal( "Expected selectedSnps.size() to be non-zero at this point." );
      }

   rgSnpInfo.reserve( selectedSnps.size() );
   for ( size_t iSelected=0; iSelected<selectedSnps.size(); ++iSelected )
      {
      size_t iSnp = selectedSnps[ iSelected ];
      rgSnpInfo.push_back( rgTPed[ iSnp ].snpInfo );
      }
   }

void CPlinkFile::ExtractSnpsFromRgMap()
   {
   if ( selectedSnps.size() == 0 )
      {
      Fatal( "Expected selectedSnps.size() to be non-zero at this point." );
      }

   rgSnpInfo.reserve( selectedSnps.size() );
   for ( size_t iSelected=0; iSelected<selectedSnps.size(); ++iSelected )
      {
      size_t iSnp = selectedSnps[ iSelected ];
      rgSnpInfo.push_back( rgMap[ iSnp ].snpInfo );
      }
   }

void CPlinkFile::ExtractSnpsFromRgMap2()
   {
   if ( selectedSnps.size() == 0 )
      {
      Fatal( "Expected selectedSnps.size() to be non-zero at this point." );
      }

   rgSnpInfo.reserve( selectedSnps.size() );
   for ( size_t iSelected=0; iSelected<selectedSnps.size(); ++iSelected )
      {
      size_t iSnp = selectedSnps[ iSelected ];

      // make sure SNP is valid across selected individuals too
      if ( FSnpHasVariation( iSnp ) )
         {
         rgSnpInfo.push_back( rgMap[ iSnp ].snpInfo );
         }
      }
   }

void CPlinkFile::FreePrivateMemory()
   {
   rgPed.clear();
   rgFam.clear();
   rgMap.clear();
   rgBinaryGenotype.clear();
   indexOfFamilyIdAndIndividualId.clear();
   indexOfSnpIds.clear();
   }

void CPlinkFile::ReadDosageFiles()
   {
   CTimer timer( true );

   /*
    *  read the individual's demographic information. 
    *  the .Fam defines the indviduals to operate over
    *  and the 'list' of individuals is in rgFam
    */
   ReadFamFile();

   /*
    *  Read the array of dosage information from the .DAT file
    *    Dosage is genotype expressed as probability of specific
    *    genotype/allele
    */
   std::string filename = baseFilename + ".dat";
   CPlinkDatFile datFile( filename );
   datFile.Load( &rgFam );

   // see if a ".map" file is present and use it to fill in the SnpInfo block
   //   and define which SNPs are to be used in the analysis
   filename = baseFilename + ".map";
   CPlinkMapFile mapFile( filename );
   if ( FFileExists( filename ) )
      {
      mapFile.Load();
      /*
       *  Now flesh out the missing SnpInfo fields in the .MAP file with
       *  values from the .DAT file.
       */
      for( size_t iMapRecord = 0; iMapRecord<mapFile.CountOfMapRecords(); ++iMapRecord )
         {
         MapRecord* pmr = &mapFile.rgMapRecords[ iMapRecord ];
         DatRecord* pdr = datFile.DatRecordPointer( pmr->snpInfo.idSnp );
         if ( pdr != nullptr )
            {
            /*
             *  Copy the allele information form the .DAT file to the 
             *    snpInfo in the mapFile
             */
            pmr->snpInfo.majorAllele = pdr->majorAllele;
            pmr->snpInfo.minorAllele = pdr->minorAllele;
            }
         AddSnpIdToIndex( pmr->snpInfo.idSnp, rgMap.size() );
         rgMap.push_back( *pmr );
         }
      }
   else
      {
      /*
       *  We don't have a .MAP file, so 'create' the rest of the SnpInfo
       *  for FastLmm
       */
      MapRecord mr;
      char *fastLmmTesting = getenv( "FastLmmTesting" );
      if ( fastLmmTesting != nullptr )
         {
         mr.snpInfo.iChromosome = 1;
         mr.snpInfo.idChromosome = "1";
         }
      for ( size_t idx=0; idx<datFile.CountOfSnps(); ++idx )
         {
         DatRecord* pdr = datFile.DatRecordPointer( idx );
         mr.snpInfo.idSnp = pdr->idSnp;
         mr.snpInfo.majorAllele = pdr->majorAllele;
         mr.snpInfo.minorAllele = pdr->minorAllele;
         AddSnpIdToIndex( mr.snpInfo.idSnp, rgMap.size() );
         rgMap.push_back( mr );
         }
      }
   /*
    *  We have all the data in memory.  Validate and organize it properly
    */
   ProduceFilteredDataFromDosageFiles( datFile, mapFile );
   snpData = snpArray;
   phenotypeData = phenArray;
   snpArray = nullptr;
   phenArray = nullptr;

   timer.Stop();
   timer.Report( 0x02, "       ReadDosageFiles elapsed time: %s" );
   }

void CPlinkFile::ProduceFilteredDataFromDosageFiles( CPlinkDatFile& datFile, CPlinkMapFile& mapFile )
   {
   //DEBUG:  Assert Phenotype info is already setup
   if ( (cPhenotypes != 1) || (ind_phenotype != 0) || (phenotypeLabels.size() != 1) )
      {
      Fatal( "Phenotype info not properly setup, expected 1, 0, 1 and found %d, %d, %d", cPhenotypes, ind_phenotype, phenotypeLabels.size() );
      }

   SelectIndividualsFromDosageFiles( alternatePhenotypeFile, datFile );   // setup individualLabels array

   cSnpsRead = datFile.rgDat.size();
   SelectSnpsFromDosageFiles( datFile );           // apply any filtering needed and then
   ExtractSnpsFromDosage( datFile, mapFile );      // we need to copy them over to snpInfo for later use too.

   CreateDosageColumnMajorSnpArray( datFile );
   CreateDosageColumnMajorPhenArray();
   }

void CPlinkFile::SelectIndividualsFromDosageFiles( CPlinkAlternatePhenotypeFile& altPhenotypeFile, CPlinkDatFile& datFile )
   {
   /*
    *  We know we have an alternatePhenotype which drives the individuals to analyze
    *  However we must have the individual information in the alternate phenotype file
    *    the FAM file, and the .DAT file or we skip the individual
    */
   size_t cAlternatePhenotypeIndividuals = alternatePhenotypeFile.CountOfIndividuals();
   individualLabels.reserve( cAlternatePhenotypeIndividuals );
   selectedIndividuals.reserve( cAlternatePhenotypeIndividuals );
   for ( size_t iIndividual=0; iIndividual < cAlternatePhenotypeIndividuals; ++iIndividual )
      {
      std::string key;
      altPhenotypeFile.KeyFromIndividualIndex( iIndividual, key );

      // verify there is phenotype associated with this record.
      limix::mfloat_t phenotype = altPhenotypeFile.PhenotypeValue( key );
      if ( phenotype != phenotype )
         {
         Warn( "Found 'missing' phenotype marker.  Excluding [%s] from the study", key.c_str() );
         continue;
         }

      // verify the individual exists in the fam/tfam file
      if ( indexOfFamilyIdAndIndividualId.count( key ) == 0 )
         {
         Warn( "Missing .Fam/.TFam information.  Excluding [%s] from the study", key.c_str() );
         continue;
         }

      // verify the individual exists in the dosage file
      if ( !datFile.FKeyIndividualInDatFile( key ) )
         {
         Warn( "Missing .dat information.  Excluding [%s} from the study", key.c_str() );
         continue;
         }

      // verify the individual exists in the covariates file (if there is one)
      if ( this->covariatesFileData && !covariatesFileData->FKeyIndividualInDatFile( key ) )
         {
         Warn( "Individual from alternate phenotype file not found in Covariates file [%s]"
             "\nExcluding [%s] from the study", covariatesFileData->filename.c_str(), key.c_str() );
         continue;
         }

      // the individual exists in all the files,
      size_t indexT = datFile.IdxFromKeyIndividual( key );
      individualLabels.push_back( key );
      selectedIndividuals.push_back( indexT );
      }

   cIndividuals = individualLabels.size();
   }

void CPlinkFile::SelectSnpsFromDosageFiles( CPlinkDatFile& datFile )
   {
   // Select the SNPs to propagate based on filtering
   //   and the individuals that are in the study sample
   //   If there is no genotype variation within the SNP
   //   in the study sample, remove the SNP from the study
   switch( snpFilter.filterType )
      {
   case SnpFilterOptions::FilterByNone:
      // No SNP filter applied.  Use all the SNPs
      selectedSnps.reserve( cSnpsRead );
      for ( size_t iSnp=0; iSnp<cSnpsRead; ++iSnp )
         {
         if ( datFile.FSnpHasVariation( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            SnpInfo snpInfo;
            datFile.FGetSnpInfo( iSnp, snpInfo );
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", snpInfo.idSnp.c_str() );
            }
         }
      break;
   case SnpFilterOptions::FilterByJob:
      {
      // No extract file, so it must be a sequential subset 
      //   created from 'Job' and JobNumber
      size_t firstSnp = snpFilter.IndexToFirstSnpInJob( cSnpsRead );
      size_t lastSnp = snpFilter.IndexToLastSnpInJob( cSnpsRead );
      selectedSnps.reserve( lastSnp - firstSnp );

      for( size_t iSnp=firstSnp; iSnp<lastSnp; ++iSnp )
         {
         if ( datFile.FSnpHasVariation( iSnp ) )
            {
            selectedSnps.push_back( iSnp );
            }
         else
            {
            Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ iSnp ].snpInfo.idSnp.c_str() );
            }
         }
      }
      break;
   case SnpFilterOptions::FilterByFilelist:
   case SnpFilterOptions::FilterByTopN:
      // match the snps in snpFilter.snpsToProcess against the snps
      //   we have the map of SnpIds to index in indexOfSnpIds
      //   we have the list of SnpIds we want in snpFilter.snpIdsToExtract
      //   get the indexes put into selectedSnps (and then sort?) to be in original order
      selectedSnps.reserve( snpFilter.snpIdsToExtract.size() );
      for ( size_t i=0; i<snpFilter.snpIdsToExtract.size(); ++i )
         {
         if ( indexOfSnpIds.count( snpFilter.snpIdsToExtract[i] ) == 0 )
            {
            Warn( "SNP not found.  Cannot extract SNP information for [%s].  Skipping...", snpFilter.snpIdsToExtract[i].c_str() );
            }
         else
            {
            size_t iSnp = indexOfSnpIds[ snpFilter.snpIdsToExtract[i] ];
            if ( datFile.FSnpHasVariation( iSnp ) )
               {
               selectedSnps.push_back( iSnp );
               }
            else
               {
               Warn( "SNP[%s] has no variation.  Filtering SNP from dataset.", rgMap[ i ].snpInfo.idSnp.c_str() );
               }
            }
         }
      sort( selectedSnps.begin(), selectedSnps.end() );
      break;
   default:
      Fatal( "Unexpected snpFilter type.  %d", snpFilter.filterType );
      }

   cSnps = selectedSnps.size();
   if ( cSnps == 0 )
      {
      Fatal( "No SNPs passed filter criteria." );
      }
   }

void CPlinkFile::ExtractSnpsFromDosage( CPlinkDatFile& datFile, CPlinkMapFile& mapFile )
   {
   ExtractSnpsFromRgMap();
   }
}// end :plink