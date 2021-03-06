﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;

namespace DevTreks.Extensions
{
    /// <summary>
    ///Purpose:		Run food nutrition stock calculations for operations and components.
    ///Author:		www.devtreks.org
    ///Date:		2017, September
    /// </summary>
    public class OCFNSR01Calculator
    {
        public OCFNSR01Calculator(CalculatorParameters calcParameters)
        {
            BIFNSR01Calculator = new BIFNSR01StockCalculator(calcParameters);
        }
        //stateful food nutrition stock
        public BIFNSR01StockCalculator BIFNSR01Calculator { get; set; }

        public bool AddCalculationsToCurrentElement(
            XElement currentCalculationsElement, XElement currentElement)
        {
            bool bHasCalculations = false;
            if (currentElement.Name.LocalName
                == OperationComponent.OPERATION_PRICE_TYPES.operationgroup.ToString()
                || currentElement.Name.LocalName
                == OperationComponent.COMPONENT_PRICE_TYPES.componentgroup.ToString())
            {
                //the operation group can be used to insert calculators into 
                //descendant operations and run totals for each operation
                bHasCalculations = BIFNSR01Calculator.SetTotalFNSR01StockCalculations(
                    currentCalculationsElement, currentElement);
            }
            else if (currentElement.Name.LocalName
                .EndsWith(OperationComponent.OPERATION_PRICE_TYPES.operation.ToString())
                || currentElement.Name.LocalName
                .EndsWith(OperationComponent.COMPONENT_PRICE_TYPES.component.ToString()))
            {
                //bifnsr01stockcalculator handles calculations
                bHasCalculations = BIFNSR01Calculator.SetOpOrCompFNSR01StockCalculations(
                    currentCalculationsElement, currentElement);
            }
            else if (currentElement.Name.LocalName
                .EndsWith(Input.INPUT_PRICE_TYPES.input.ToString()))
            {
                //resource stock calcs come from calculator results
                if (currentCalculationsElement != null)
                {
                    bHasCalculations = BIFNSR01Calculator.SetInputFNSR01StockCalculations(
                        currentCalculationsElement, currentElement);
                }
            }
            return bHasCalculations;
        }
    }
}
