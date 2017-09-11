﻿<?xml version="1.0" encoding="UTF-8" ?>
<!-- Author: www.devtreks.org, 2012, October -->
<xsl:stylesheet version="1.0"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns:DisplayDevPacks="urn:displaydevpacks">
	<xsl:output method="xml" indent="yes" omit-xml-declaration="yes" encoding="UTF-8" />
	<!-- pass in params -->
	<!--array holding references to constants, locals, lists ...-->
	<xsl:param name="linkedListsArray" />
	<!-- calcs -->
	<xsl:param name="saveMethod" />
	<!-- default linked view -->
	<xsl:param name="defaultLinkedViewId" />
	<!-- the last step is used to display different parts of the calculator-->
	<xsl:param name="lastStepNumber" />
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
	<!-- the node being calculated (custom docs' nodename can be a devpack node, while this might be budgetgroup) -->
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
	<!-- what is the club's email? -->
	<xsl:param name="clubEmail" />
	<!-- what is the current serviceid? -->
	<xsl:param name="contenturipattern" />
	<!-- path to resources -->
	<xsl:param name="fullFilePath" />
	<!-- init html -->
	<xsl:template match="@*|/|node()" />
	<xsl:template match="/">
		<div class="box">
			<xsl:if test="($serverActionType = 'linkedviews')">
				<div id="stepsmenu">
					<xsl:value-of select="DisplayDevPacks:WriteMenuSteps('3')"/>
				</div>
			</xsl:if>
			<xsl:apply-templates select="root" />
		</div>
	</xsl:template>
	<xsl:template match="root">
		<xsl:variable name="linkedviewid"><xsl:value-of select="DisplayDevPacks:GetURIPatternPart($calcDocURI,'id')" /></xsl:variable>
		<div id="divstepzero">
      <h4 class="ui-bar-b"><strong>Output Calculation View</strong></h4>
      <xsl:choose>
        <xsl:when test="($docToCalcNodeName = 'outputgroup' 
							or $docToCalcNodeName = 'output' 
							or $docToCalcNodeName = 'outputseries'
							or contains($docToCalcNodeName, 'devpack') 
							or contains($docToCalcNodeName, 'linkedview'))">
				</xsl:when>
				<xsl:otherwise>
					<h3>This output calculator does not appear appropriate for the document being analyzed. Are you 
					sure this is the right calculator?</h3>
				</xsl:otherwise>
			</xsl:choose>
      <h4 class="ui-bar-b">
					<strong>
						<xsl:value-of select="linkedview[@Id=$linkedviewid]/@CalculatorName" />
					</strong>
			</h4>
			<p>
					<strong>Introduction</strong>
					<br />
					This calculator calculates discounted output totals.
			</p>
			<p>
				<strong>Calculation View Description</strong>
				<br />
				<xsl:value-of select="linkedview[@Id=$linkedviewid]/@CalculatorDescription" />
			</p>
			<p>
				<strong>Version: </strong>
				<xsl:value-of select="linkedview[@Id=$linkedviewid]/@Version"/>
			</p>
			<p>
				<a id="aFeedback" name="Feedback">
					<xsl:attribute name="href">mailto:<xsl:value-of select="$clubEmail" />?subject=<xsl:value-of select="$selectedFileURIPattern" /></xsl:attribute>
					Feedback About <xsl:value-of select="$selectedFileURIPattern" />
				</a>
			</p>
		</div>
		<xsl:apply-templates select="linkedview" />
	</xsl:template>
	<xsl:template match="linkedview">
		<xsl:variable name="linkedviewid"><xsl:value-of select="DisplayDevPacks:GetURIPatternPart($calcDocURI,'id')" /></xsl:variable>
		<xsl:if test="(@Id = $linkedviewid) or (@Id = 1)">
		<xsl:variable name="searchurl"><xsl:value-of select="DisplayDevPacks:GetURIPattern(@CalculatorName,@Id,$networkId,local-name(),'')" /></xsl:variable>
		<div id="divstepone">
      <h4 class="ui-bar-b"><strong>Step 1 of 3. Make Selections</strong></h4>
		  <xsl:variable name="calcParams1">'&amp;step=steptwo<xsl:value-of select="$calcParams" />'</xsl:variable>
			<xsl:if test="($viewEditType = 'full')">
				<xsl:value-of select="DisplayDevPacks:WriteGetConstantsButtons($selectedFileURIPattern, $contenturipattern, $calcParams1)"/>
			</xsl:if>
			<xsl:if test="($lastStepNumber = 'steptwo')">
        <h4 class="ui-bar-b"><strong>Success. Please review the selections below.</strong></h4>
			</xsl:if>
      <br />
			<xsl:value-of select="DisplayDevPacks:WriteSelectListsForLocals(
						$linkedListsArray, $searchurl, $serverSubActionType, 'full', 
						@RealRate, @NominalRate, @UnitGroupId, @CurrencyGroupId,
						@RealRateId, @NominalRateId, @RatingGroupId)"/>
		</div>
    <div id="divsteptwo">
			<h4 class="ui-bar-b"><strong>Step 2 of 3. Calculate</strong></h4>
			<xsl:variable name="calcParams3">'&amp;step=stepthree<xsl:value-of select="$calcParams"/>'</xsl:variable>
      <xsl:if test="($viewEditType = 'full')">
				<xsl:value-of select="DisplayDevPacks:WriteCalculateButtons($selectedFileURIPattern, $contenturipattern, $calcParams3)"/>
			</xsl:if>
			<xsl:if test="($lastStepNumber = 'stepthree')">
          <h4 class="ui-bar-b"><strong>Success. Please review the calculations below.</strong></h4>
			</xsl:if>
      <div data-role="collapsible"  data-theme="b" data-content-theme="d">
      <h4>Relations</h4>
	    <xsl:if test="(($docToCalcNodeName != 'outputseries') or contains($docToCalcNodeName, 'devpack') or contains($docToCalcNodeName, 'linkedview'))">
				<xsl:value-of select="DisplayDevPacks:WriteStandardCalculatorParams1($searchurl, $viewEditType,
          @UseSameCalculator, @Overwrite)"/>
		  </xsl:if>
      <xsl:value-of select="DisplayDevPacks:WriteStandardCalculatorParams2($searchurl, $viewEditType,
          @WhatIfTagName, @RelatedCalculatorsType)"/>
      </div>
				<xsl:if test="($lastStepNumber = 'stepthree') 
						or ($viewEditType != 'full')
						or (@TR >= 0)">
          <div data-role="collapsible"  data-theme="b" data-content-theme="d">
					<div class="ui-grid-a">
						<div class="ui-block-a">
							<strong>Output Amount</strong>: 
							<xsl:value-of select="@OutputAmount1" />
						</div>
						<div class="ui-block-b">
							<strong>Output Unit</strong>: 
							<xsl:value-of select="@OutputUnit1" />
						</div>
					</div>
					<div class="ui-grid-a">
						<div class="ui-block-a">
							<strong>Output Price</strong>: 
							<xsl:value-of select="@OutputPrice1" />
						</div>
						<div class="ui-block-b">
							<strong>Output Date Received</strong>: 
							<xsl:value-of select="@OutputDate" />
						</div>
					</div>
					<div class="ui-grid-a">
						<div class="ui-block-a">
							<strong>Composition Amount</strong>: 
							<xsl:value-of select="@OutputCompositionAmount" />
						</div>
						<div class="ui-block-b">
							<strong>Composition Unit</strong>: 
							<xsl:value-of select="@OutputCompositionUnit" />
						</div>
					</div>
					<div class="ui-grid-a">
						<div class="ui-block-a">
							<strong>Total Revenue</strong>: 
							<xsl:value-of select="@TR" />
						</div>
						<div class="ui-block-b">
							<strong>Interest</strong>: 
							<xsl:value-of select="@TR_INT" />
						</div>
					</div>
					<div class="ui-grid-a">
            <div class="ui-block-a">
							<strong>Discounted Revenue</strong>: 
							<xsl:value-of select="format-number((@TR + @TR_INT), '#,###.00')"/>
						</div>
					<div class="ui-block-b">
							<strong>Incentive Adjusted Total Revenues</strong>: 
							<xsl:value-of select="@TRINCENT" />
						</div>
					</div>
          </div>
				</xsl:if>
					<div>
            <label for="CalculatorName" class="ui-hidden-accessible"></label>
				    <input id="CalculatorName" type="text" data-mini="true">
              <xsl:if test="($viewEditType = 'full')">
						    <xsl:attribute name="name"><xsl:value-of select="$searchurl" />;CalculatorName;string;75</xsl:attribute>
              </xsl:if>
					    <xsl:attribute name="value"><xsl:value-of select="@CalculatorName" /></xsl:attribute>
				    </input>
		    </div>
				<xsl:if test="($docToCalcNodeName = 'outputgroup' or contains($docToCalcNodeName, 'linkedview'))">
					<div class="ui-grid-a">
						<div class="ui-block-a">
							<label for="lblOutputPrice">Output Price</label>
							<input type="text" id="lblOutputPrice">
                <xsl:if test="($viewEditType = 'full')">
									<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;OutputPrice1;decimal;8</xsl:attribute>
								</xsl:if>
                <xsl:attribute name="value"><xsl:value-of select="@OutputPrice1" /></xsl:attribute>
							</input>
						</div>
						<div class="ui-block-b">
							<label for="lblOutputAmount">Output Amount</label>
							<input type="text" id="lblOutputAmount">
                <xsl:if test="($viewEditType = 'full')">
									<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;OutputAmount1;double;8</xsl:attribute>
								</xsl:if>
                <xsl:attribute name="value"><xsl:value-of select="@OutputAmount1" /></xsl:attribute>
							</input>
						</div>
					</div>
					<div class="ui-grid-a">
						<div class="ui-block-a">
							<label for="lblOutputDate">Output Date</label>
							<input type="text" id="lblOutputDate">
                <xsl:if test="($viewEditType = 'full')">
									<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;OutputDate;datetime;8</xsl:attribute>
								</xsl:if>
                <xsl:attribute name="value"><xsl:value-of select="@OutputDate" /></xsl:attribute>
							</input>
						</div>
						<div class="ui-block-b">
							<label for="lblOutputUnit">Output Unit</label>
							<select class="SelectUnits" id="lblOutputUnit" data-mini="true">
                <xsl:if test="($viewEditType = 'full')">
									<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;OutputUnit1;string;75</xsl:attribute>
								</xsl:if>
                <xsl:attribute name="data-unitid"><xsl:value-of select="@UnitGroupId" /></xsl:attribute>
								<option>
									<xsl:attribute name="value"><xsl:value-of select="@OutputUnit1" /></xsl:attribute>
									<xsl:attribute name="selected" />
									<xsl:value-of select="@OutputUnit1" />
								</option>
							</select>
						</div>
					</div>
				</xsl:if>
				<div class="ui-grid-a">
					<div class="ui-block-a">
						<label for="lblOutputCompositionAmount">Composition Amount</label>
						<input type="text" id="lblOutputCompositionAmount">
              <xsl:if test="($viewEditType = 'full')">
								<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;OutputCompositionAmount;double;8</xsl:attribute>
							</xsl:if>
              <xsl:attribute name="value"><xsl:value-of select="@OutputCompositionAmount" /></xsl:attribute>
						</input>
					</div>
					<div class="ui-block-b">
						<label for="lblOutputCompositionUnit">Composition Unit</label>
						<select class="SelectUnits" id="lblOutputCompositionUnit" data-mini="true">
              <xsl:if test="($viewEditType = 'full')">
								<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;OutputCompositionUnit;string;25</xsl:attribute>
							</xsl:if>
              <xsl:attribute name="data-unitid"><xsl:value-of select="@UnitGroupId" /></xsl:attribute>
							<option>
								<xsl:attribute name="value"><xsl:value-of select="@OutputCompositionUnit" /></xsl:attribute>
								<xsl:attribute name="selected" />
								<xsl:value-of select="@OutputCompositionUnit" />
							</option>
						</select>
					</div>
				</div>
				<div class="ui-field-contain">
          <fieldset data-role="controlgroup" data-type="horizontal" data-mini="true">
            <legend>Is Discounted?</legend>
            <input type="radio" id="lblDiscountYorN1">
              <xsl:if test="($viewEditType = 'full')">
                <xsl:attribute name="name"><xsl:value-of select="$searchurl" />;DiscountYorN;boolean;1</xsl:attribute>
              </xsl:if>
              <xsl:attribute name="value">1</xsl:attribute>
              <xsl:if test="(@DiscountYorN = '1')">
                <xsl:attribute name="checked">true</xsl:attribute>
              </xsl:if>
            </input>
            <label for="lblDiscountYorN1">True</label>
            <input type="radio" id="lblDiscountYorN2">
              <xsl:if test="($viewEditType = 'full')">
                <xsl:attribute name="name"><xsl:value-of select="$searchurl" />;DiscountYorN;boolean;1</xsl:attribute>
              </xsl:if>
              <xsl:attribute name="value">0</xsl:attribute>
              <xsl:if test="(@DiscountYorN != '1')">
                <xsl:attribute name="checked">true</xsl:attribute>
              </xsl:if>
            </input>
            <label for="lblDiscountYorN2">False</label>
          </fieldset>
				</div>
        <div>
						<label for="lblEOPDate" >End of Period Date</label>
						<input type="text" id="lblEOPDate">
							<xsl:if test="($viewEditType = 'full')">
                <xsl:attribute name="name"><xsl:value-of select="$searchurl" />;Date;datetime;8</xsl:attribute>
							</xsl:if>
              <xsl:attribute name="value"><xsl:value-of select="@Date" /></xsl:attribute>
						</input>
				</div>
				<div class="ui-grid-a">
					<div class="ui-block-a">
						<label for="lblOutputIncentiveAmount" >Incentive Amount</label>
						<input id="lblOutputIncentiveAmount">
              <xsl:if test="($viewEditType = 'full')">
								<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;IncentiveAmount;double;8</xsl:attribute>
							</xsl:if>
              <xsl:attribute name="value"><xsl:value-of select="DisplayDevPacks:WriteFormattedNumber('IncentiveAmount', @IncentiveAmount, 'double', '8')"/></xsl:attribute>
						</input>
					</div>
					<div class="ui-block-b">
						<label for="lblOutputIncentiveRate" >Incentive Rate</label>
							<input id="lblOutputIncentiveRate">
                <xsl:if test="($viewEditType = 'full')">
									<xsl:attribute name="name"><xsl:value-of select="$searchurl" />;IncentiveRate;double;8</xsl:attribute>
								</xsl:if>
                <xsl:attribute name="value"><xsl:value-of select="DisplayDevPacks:WriteFormattedNumber('IncentiveRate', @IncentiveRate, 'double', '8')"/></xsl:attribute>
							</input>
					</div>
				</div>
				<div >
				  <label for="lblDescription">Description</label>
				  <textarea class="Text75H100PCW" id="lblDescription" data-mini="true">
            <xsl:if test="($viewEditType = 'full')">
						  <xsl:attribute name="name"><xsl:value-of select="$searchurl" />;CalculatorDescription;string;255</xsl:attribute>
            </xsl:if>
					  <xsl:value-of select="@CalculatorDescription" />
				  </textarea>
			  </div>
        <div >
				  <label for="lblMediaURL">Media URL</label>
				  <textarea class="Text75H100PCW" id="lblMediaURL" data-mini="true">
            <xsl:if test="($viewEditType = 'full')">
						  <xsl:attribute name="name"><xsl:value-of select="$searchurl" />;MediaURL;string;500</xsl:attribute>
            </xsl:if>
					  <xsl:value-of select="@MediaURL" />
				  </textarea>
			  </div>
    </div>
		<div id="divsteplast">
			<h4 class="ui-bar-b"><strong>Instructions (beta)</strong></h4>
      <div data-role="collapsible"  data-theme="b" data-content-theme="d">
			<h4>Step 1</h4>
			<ul data-role="listview">
				<li><strong>Step 1. Real and Nominal Rates:</strong> The nominal rate includes inflation while the real rate does not. In the USA, DevTreks recommends 
					using Office of Management and Budget rates for the same year as the date of the input.</li>
			</ul>
			</div>
      <div data-role="collapsible"  data-theme="b" data-content-theme="d">
			<h4>Step 2</h4>
			<ul data-role="listview">
				<li><strong>Step 2. Use Same Calculator Pack In Descendants?:</strong> True to insert or update this same calculator in children.</li>
				<li><strong>Step 2. Overwrite Descendants?:</strong> True to insert or update all of the attributes of this same calculator in all children. False only updates children attributes that are controlled by the developer of this calculator (i.e. version, stylehsheet name, relatedcalculatorstype ...)</li>
        <li><strong>Step 2. What If Tag Name:</strong> Instructional videos explaining the use of what-if scenario calculations should be viewed before changing or using this parameter.</li>
				<li><strong>Step 2. Related Calculators Type:</strong> When the Use Same Calculator Pack in Descendant is true, uses this value to determine which descendant calculator to update. Inserts a new descendant when no descendant has this same name. Updates the descendant that has this same name.</li>
				<li><strong>Step 2. Output Variables:</strong> These are the same output variables found in operating and capital budgets and are used in the same manner.</li>
			</ul>
      </div>
		</div>
	</xsl:if>
	</xsl:template>
</xsl:stylesheet>
