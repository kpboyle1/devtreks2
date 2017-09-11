﻿<?xml version="1.0" encoding="UTF-8" ?>
<!-- Author: www.devtreks.org, 2012, July -->
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
	xmlns:y0="urn:DevTreks-support-schemas:Investment"
	xmlns:DisplayDevPacks="urn:displaydevpacks">
	<xsl:output method="xml" indent="yes" omit-xml-declaration="yes" encoding="UTF-8" />
	<!-- pass in params -->
	<!-- what action is being taken by the server -->
	<xsl:param name="serverActionType" />
	<!-- what other action is being taken by the server -->
	<xsl:param name="serverSubActionType" />
	<!-- is the member viewing this uri the owner? -->
	<xsl:param name="isURIOwningClub" />
	<!-- which node to start with? -->
	<xsl:param name="nodeName" />
	<!-- which view to use? -->
	<xsl:param name="viewEditType" />
	<!-- is this a coordinator? -->
	<xsl:param name="memberRole" />
	<!-- what is the current uri? -->
	<xsl:param name="selectedFileURIPattern" />
	<!-- the addin being used -->
	<xsl:param name="calcDocURI" />
	<!-- the node being calculated-->
	<xsl:param name="docToCalcNodeName" />
	<!-- standard params used with calcs and custom docs -->
	<xsl:param name="calcParams" />
	<!-- what is the name of the node to be selected? -->
	<xsl:param name="selectionsNodeNeededName" />
	<!-- which network is this doc from? -->
	<xsl:param name="networkId" />
	<!-- what is the start row? -->
	<xsl:param name="startRow" />
	<!-- what is the end row? -->
	<xsl:param name="endRow" />
	<!-- what are the pagination properties ? -->
	<xsl:param name="pageParams" />
	<!-- what is the guide's email? -->
	<xsl:param name="clubEmail" />
	<!-- what is the current serviceid? -->
	<xsl:param name="contenturipattern" />
	<!-- path to resources -->
	<xsl:param name="fullFilePath" />
	<!-- init html -->
	<xsl:template match="@*|/|node()" />
	<xsl:template match="/">
		<xsl:apply-templates select="root" />
	</xsl:template>
	<xsl:template match="root">
		<div id="modEdits_divEditsDoc">
			<table class="data" cellpadding="6" cellspacing="1" border="0">
				<tbody>
					<xsl:apply-templates select="y0:servicebase" />
					<xsl:apply-templates select="servicebase" />
					<xsl:apply-templates select="investmentgroup" />
					<tr id="footer">
						<td scope="row" colspan="10">
							<a id="aFeedback" name="Feedback">
								<xsl:attribute name="href">mailto:<xsl:value-of select="$clubEmail" />?subject=<xsl:value-of select="$selectedFileURIPattern" /></xsl:attribute>
								Feedback About <xsl:value-of select="$selectedFileURIPattern" />
							</a>
						</td>
					</tr>
				</tbody>
			</table>
		</div>
	</xsl:template>
	<xsl:template match="y0:servicebase">
		<tr>
			<th colspan="10">
				Service: <xsl:value-of select="@Name" />
			</th>
		</tr>
		<xsl:apply-templates select="investmentgroup">
			<xsl:sort select="@Name"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="servicebase">
		<tr>
			<th colspan="10">
				Service: <xsl:value-of select="@Name" />
			</th>
		</tr>
		<xsl:apply-templates select="investmentgroup">
			<xsl:sort select="@Name"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investmentgroup">
		<tr>
			<th scope="col" colspan="10">
				Investment Group
			</th>
		</tr>
		<tr>
			<td colspan="10">
				<strong><xsl:value-of select="@Name" /> </strong>
			</td>
		</tr>
		<xsl:apply-templates select="investment">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
		<tr>
			<td colspan="10"><strong>Group Totals</strong></td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investment">
		<tr>
			<th scope="col" colspan="10">
				Investment
			</th>
		</tr>
		<tr>
			<td colspan="10">
				<strong><xsl:value-of select="@Name" /> </strong>
			</td>
		</tr>
		<xsl:apply-templates select="investmenttimeperiod">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
		<tr>
			<td colspan="10"><strong>Investment Totals</strong></td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investmenttimeperiod">
		<tr>
			<th scope="col" colspan="10"><strong>Time Period</strong></th>
		</tr>
		<tr>
			<td scope="row" colspan="10">
				<strong><xsl:value-of select="@PracticeName" />(Amount:<xsl:value-of select="@PracticeAmount" />Unit:<xsl:value-of select="@PracticeUnit" />)</strong>
			</td>
		</tr>
		<xsl:variable name="outputCount" select="count(investmentoutcomes/investmentoutcome)"/>
		<xsl:if test="($outputCount != '0')">
			<tr>
				<th scope="col" colspan="10">Outcomes</th>
			</tr>
		  <tr>
			  <td>
				  <strong>
					  <label for="lblOutputName" >Output Name</label>
				  </strong>
			  </td>
        <td>
				  <strong>
					  <label for="lblOutputDate" >Date</label>
				  </strong>
			  </td>
			  <td>
				  <strong>
					  <label for="lblOutputUnit" >Unit</label>
				  </strong>
			  </td>
			  <td>
				  <strong>
					  <label for="lblOutputPrice" >Price</label>
				  </strong>
			  </td>
			  <td>
				  <strong>
					  <label for="lblOutputYield" >Yield</label>
				  </strong>
			  </td>
			  <td>
				  <strong>
					  <label for="lblCompositionAmount" >Composition Amount</label>
				  </strong>
			  </td>
			  <td>
				  <strong>
					  <label for="lblCompositionUnit" >Composition Unit</label>
				  </strong>
			  </td>
			  <td>
				  <strong>
					  <label for="lblTotalRevenue" >Total Revenue</label>
				  </strong>
			  </td>
			  <td>
				  <strong>
					  <label for="lblAnnualTotal" >Annual Total</label>
				  </strong>
			  </td>
        <td>
				  <strong>
					  <label for="" ></label>
				  </strong>
			  </td>
		  </tr>
			<xsl:apply-templates select="investmentoutcomes" />
		</xsl:if>
		<xsl:apply-templates select="investmentcomponents" />
		<tr>
			<td colspan="10"><strong>Time Period Totals</strong></td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investmentoutcomes">
		<xsl:apply-templates select="investmentoutcome">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investmentoutcome">
		<tr>
			<th scope="col" colspan="10"><strong>Outcome</strong></th>
		</tr>
		<tr>
			<td scope="row" colspan="10">
				<strong><xsl:value-of select="@Name" /></strong>
			</td>
		</tr>
    <tr>
			<th scope="col" colspan="10"><strong>Outputs</strong></th>
		</tr>
		<xsl:apply-templates select="investmentoutput">
    </xsl:apply-templates>
		<tr>
			<td colspan="10"><strong>Outcome Totals&#xA0;&#xA0;(Amount:&#xA0;<xsl:value-of select="@Amount" />;&#xA0;Date:&#xA0;<xsl:value-of select="@Date" /></strong></td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investmentoutput">
		<tr>
			<td>
				<xsl:value-of select="@Name" />
			</td>
			<td>
				<xsl:value-of select="@OutputDate" />
			</td>
			<td>
				<xsl:value-of select="@OutputUnit1" />
			</td>
			<td>
				<xsl:value-of select="@OutputPrice1" />
			</td>
      <td>
				<xsl:value-of select="@OutputAmount1"/>
			</td>
			<td>
				<xsl:value-of select="@OutputCompositionAmount" />
			</td>
			<td>
				<xsl:value-of select="@OutputCompositionUnit"/>
			</td>
			<td>
				<xsl:value-of select="@TR" />
			</td>
			<td>
				<xsl:value-of select="@TAMR" />
			</td>
      <td>
				
			</td>
		</tr>
	</xsl:template>
	<xsl:template match="investmentcomponents">
		<xsl:apply-templates select="investmentcomponent">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investmentcomponent">
		<tr>
			<th scope="col" colspan="10"><strong>Component</strong></th>
		</tr>
		<tr>
			<td scope="row" colspan="10">
				<strong><xsl:value-of select="@Name" />(Amount:<xsl:value-of select="@Amount" />Unit:<xsl:value-of select="@Unit" />)</strong>
			</td>
		</tr>
    <tr>
				<th scope="col" colspan="10">Inputs (cost = cost * time.amount * component.amount * input.times * input.amount)</th>
			</tr>
      <tr>
			<td>
				<strong>
					<label for="lblMarketValue" >Market Value</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lbHPPTOMax" >Horsepower (MaxPTO)</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lbHPPTOEquiv" >Equivalent PTO HP</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblFieldSpeedTypical" >Speed</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblWidth" >Width</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblFieldEffTypical" >Field Efficiency</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblCapitalRecoveryCost" >Capital Recovery Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTaxesHousingInsuranceCost" >THI Cost</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblLubeOilCost" >Lube Oil Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblRepairCost" >Repair Cost</label>
				</strong>
			</td>
		</tr>
		<tr>
      <td>
				<strong>
					<label for="lblLaborCost" >Labor Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblFuelAmount" >Fuel Amount</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblFuelCost" >Fuel Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblOCAmount" >OC Amount</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblOCUnit" >OC Units</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblOCPrice" >OC Price</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTOC" >OC Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblAOHAmount" >AOH Amount</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblAOHPrice" >AOH Price</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTAOH" >AOH Cost</label>
				</strong>
			</td>
		</tr>
		<xsl:apply-templates select="investmentinput">
      <xsl:sort select="@InputDate"/>
    </xsl:apply-templates>
		<tr>
			<td colspan="10"><strong>Component Totals</strong></td>
		</tr>
    <tr>
			<td>
				<strong>
					<label for="lblMarketValue" >Market Value</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblCapitalRecoveryCost" >Capital Recovery Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTaxesHousingInsuranceCost" >THI Cost</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblAOHCost" >Alloc. OH Cost</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblLubeOilCost" >Lube Oil Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblRepairCost" >Repair Cost</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblLaborCost" >Labor Cost</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblFuelAmount" >Fuel Amount</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblFuelCost" >Fuel Cost</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblOCCost" >Operating Cost</label>
				</strong>
			</td>
		</tr>
		<tr>
			<td>
				<strong>
					<label for="lbLaborAvailable" >Labor Available (hours per day)</label>
				</strong>
			</td>
      <td>
				<strong>
					<label for="lblAreaCovered" >Area Covered (ac/ha per day)</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblPlannedStartDate" >Planned vs Actual Start Date</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblProbableFieldDays" >Probable Field Days Needed</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblProbableFinishDate" >Probable Finish Date</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTimelinessPenaltyDaysFromStart1" >Timeliness Penalty Days From Start</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTimelinessPenalty1" >Timeliness Penalty (percent)</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTimelinessPenalty2" >Additional Penalty (percent)</label>
				</strong>
			</td>
			<td>
				<strong>
					<label for="lblTimelinessPenaltyCost" >Timeliness Penalty Cost (currency)</label>
				</strong>
       </td>
      <td>
				<strong>
					<label for="lblTimelinessPenaltyCostPerHour" >Timeliness Penalty Cost Per Hour</label>
				</strong>
       </td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="investmentinput">
    <tr>
			<td scope="row" colspan="10">
				<strong><xsl:value-of select="@Name" />(Times:<xsl:value-of select="@InputTimes" />Date:<xsl:value-of select="@InputDate" />)</strong>
			</td>
		</tr>
	  <xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="root/linkedview">
		<xsl:param name="localName" />
    <xsl:choose>
      <xsl:when test="($localName = 'investmentcomponent')">
			<tr>
				<td>
					<xsl:value-of select="@TMarketValue" />
				</td>
        <td>
					<xsl:value-of select="@TCapitalRecoveryCost" />
				</td>
				<td>
					<xsl:value-of select="@TTaxesHousingInsuranceCost" />
				</td>
        <td>
					<xsl:value-of select="@TAOH" />
				</td>
				<td>
					<xsl:value-of select="@TLubeOilCost" />
				</td>
				<td>
					<xsl:value-of select="@TRepairCost" />
				</td>
         <td>
					<xsl:value-of select="@TLaborCost" />
				</td>
				<td>
					<xsl:value-of select="@TFuelAmount" />
				</td>
				<td>
					<xsl:value-of select="@TFuelCost" />
				</td>
        <td>
					<xsl:value-of select="@TOC" />
				</td>
			</tr>
			<tr>
       <td>
					<xsl:value-of select="@LaborAvailable" />
				</td>
				<td>
					<xsl:value-of select="@AreaCovered" />
				</td>
        <td>
					<xsl:value-of select="@PlannedStartDate" />
          ---<xsl:value-of select="@ActualStartDate" />
				</td>
				<td>
					<xsl:value-of select="@ProbableFieldDays" />
				</td>
				<td>
					<xsl:value-of select="@ProbableFinishDate" />
				</td>
				<td>
					<xsl:value-of select="@TimelinessPenaltyDaysFromStart1" />
				</td>
				<td>
					<xsl:value-of select="@TimelinessPenalty1" />
				</td>
        <td>
					<xsl:value-of select="@TimelinessPenalty2" />
				</td>
				<td>
					<xsl:value-of select="@TimelinessPenaltyCost" />
				</td>
        <td>
					<xsl:value-of select="@TimelinessPenaltyCostPerHour" />
				</td>
			</tr>
		</xsl:when>
    <xsl:when test="($localName = 'investmentinput')">
			<tr>
				<td>
					<xsl:value-of select="@MarketValue" />
				</td>
        <td>
					<xsl:value-of select="@HPPTOMax" />
				</td>
        <td>
					<xsl:value-of select="@HPPTOEquiv" />
				</td>
        <td>
					<xsl:value-of select="@FieldSpeedTypical" />
				</td>
				<td>
					<xsl:value-of select="@Width" />
				</td>
        <td>
					<xsl:value-of select="@FieldEffTypical" />
				</td>
				<td>
					<xsl:value-of select="@CapitalRecoveryCost" />
				</td>
				<td>
					<xsl:value-of select="@TaxesHousingInsuranceCost" />
				</td>
				<td>
					<xsl:value-of select="@LubeOilCost" />
				</td>
				<td>
					<xsl:value-of select="@RepairCost" />
				</td>
			</tr>
			<tr>
        <td>
					<xsl:value-of select="@LaborCost" />
				</td>
				<td>
					<xsl:value-of select="@FuelAmount" />
				</td>
				<td>
					<xsl:value-of select="@FuelCost" />
				</td>
				<td>
				<xsl:value-of select="DisplayDevPacks:WriteFormattedNumber('OCAmount', @InputPrice1Amount, 'double', '8')"/>
			  </td>
			  <td>
				  <xsl:value-of select="@InputUnit1" />
			  </td>
			  <td>
				  <xsl:value-of select="@InputPrice1" />
			  </td>
			  <td>
				  <xsl:value-of select="@TOC"/>
			  </td>
			  <td>
				  <xsl:value-of select="DisplayDevPacks:WriteFormattedNumber('AOHAmount', @InputPrice2Amount, 'double', '8')"/>
			  </td>
			  <td>
				  <xsl:value-of select="@InputPrice2" />
			  </td>
			  <td>
				  <xsl:value-of select="@TAOH"/>
			  </td>
			</tr>
		</xsl:when>
    <xsl:otherwise>
      <tr>
				<td>
					<xsl:value-of select="@TMarketValue" />
				</td>
        <td>
					<xsl:value-of select="@TCapitalRecoveryCost" />
				</td>
				<td>
					<xsl:value-of select="@TTaxesHousingInsuranceCost" />
				</td>
        <td>
					<xsl:value-of select="@TAOH" />
				</td>
				<td>
					<xsl:value-of select="@TLubeOilCost" />
				</td>
				<td>
					<xsl:value-of select="@TRepairCost" />
				</td>
         <td>
					<xsl:value-of select="@TLaborCost" />
				</td>
				<td>
					<xsl:value-of select="@TFuelAmount" />
				</td>
				<td>
					<xsl:value-of select="@TFuelCost" />
				</td>
        <td>
					<xsl:value-of select="@TOC" />
				</td>
			</tr>
			<tr>
				<td>
					<xsl:value-of select="@TLaborAvailable" />
				</td>
				<td>
					<xsl:value-of select="@TAreaCovered" />
				</td>
				<td>
					
				</td>
				<td>
					<xsl:value-of select="@TProbableFieldDays" />
				</td>
				<td>
					
				</td>
				<td>
					<xsl:value-of select="@TTimelinessPenaltyDaysFromStart1" />
				</td>
				<td>
					<xsl:value-of select="@TTimelinessPenalty1" />
				</td>
				<td>
					<xsl:value-of select="@TTimelinessPenalty2" />
				</td>
				<td>
					<xsl:value-of select="@TTimelinessPenaltyCost" />
				</td>
        <td>
					<xsl:value-of select="@TTimelinessPenaltyCostPerHour" />
				</td>
			</tr>
      </xsl:otherwise>
    </xsl:choose>
	</xsl:template>
</xsl:stylesheet>
