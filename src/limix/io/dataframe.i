namespace limix {
//CMemDataFram

%ignore CMemDataFrame::getMatrix;
%ignore CMemDataFrame::getColHeader;
%ignore CMemDataFrame::getRowHeader;

%rename(getMatrix) CMemDataFrame::agetMatrix;
%rename(getRowHeader) CMemDataFrame::aetRowHeader;
%rename(getColHeader) CMemDataFrame::aetColHeader;
}

//raw include
%include "limix/io/dataframe.h"

namespace limix{

//CRMemDataFrameXd
%template(CRMemDataFrameXd) CRMemDataFrame<MatrixXd>;
%template(CRWMemDataFrameXd) CRWMemDataFrame<MatrixXd>;
%template(ARDataFrameXd) ARDataFrame< MatrixXd >;
%template(AWDataFrameXd) AWDataFrame< MatrixXd >;

}