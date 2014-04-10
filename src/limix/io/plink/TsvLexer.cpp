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
 * TsvLexer - {Tab Separated Lexer for FastLmmC}
 *
 *         File Name:   TsvLexer.cpp [Tab Separated Values Lexer]
 *           Version:   1.00
 *            Author:   
 *     Creation Date:   18 Nov 2010
 *     Revision Date:   18 Nov 2010
 *
 *    Module Purpose:   This file implements the TSV scanner for FastLmmC
 *
 *    Change History:   
 *
 * Test Files: 
 */

/*
 * Include Files
 */
#include "TsvLexer.h"

/*
 * Externs we need
 */

/*
 * Globals we define
 */

/*
 * The code
 */
namespace plink {

	CTsvLexer::CTsvLexer(std::string filename) : CLexer(filename)
	{
		// CLexer does all the initialization
	}

	CTsvLexer::~CTsvLexer()
	{
		// CLexer does all the cleanup right now
	}


	unsigned CTsvLexer::NextToken(CToken& tok)
	{
		tok.type = tokError;
		tok.column = CurColumn();     // get the current position so we start collecting the token --  m_column;
		tok.line = CurLine();
		tok.offset = LexerFileOffset();
		tok.text.clear();

		switch (LookAheadChar())
		{
		case EOF:      // return EOF token
			tok.type = tokEOF;
			break;
		case '\t':     // return a TAB token
			tok.type = tokTab;
			AdvanceCh();
			break;
		case '\n':     // return a EOL token
			tok.type = tokEOL;
			AdvanceCh();
			break;
		default:       // see if it is a Symbol (ascii printable)
			if ((LookAheadChar() < ' ') || (LookAheadChar() > '~'))
			{
				Fatal("Found invalid character in file [%s] at Line: %d Column: %d", Filename().c_str(), tok.line, tok.column);
				break;
			}

			// accumulate ascii printable characters as a symbol
			while ((LookAheadChar() >= ' ') && (LookAheadChar() <= '~'))
			{
				tok.text += (char)LookAheadChar();
				AdvanceCh();
			}
			tok.type = tokSymbol;
			break;
		}
		return(tok.type);
	}

}//end :plink