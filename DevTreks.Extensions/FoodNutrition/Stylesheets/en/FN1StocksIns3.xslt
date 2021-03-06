﻿<?xml version="1.0" encoding="UTF-8" ?>
<!-- Author: www.devtreks.org, 2013, January -->
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
	xmlns:y0="urn:DevTreks-support-schemas:Input"
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
		<div id="mainwrapper">
			<table class="data" cellpadding="6" cellspacing="1" border="0">
				<tbody>
					<xsl:apply-templates select="servicebase" />
					<xsl:apply-templates select="inputgroup" />
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
	<xsl:template match="servicebase">
		<tr>
			<th colspan="10">
				Service: <xsl:value-of select="@Name" />
			</th>
		</tr>
		<xsl:apply-templates select="inputgroup">
			<xsl:sort select="@Name"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="inputgroup">
		<tr>
			<th scope="col" colspan="10">
				Input Group
			</th>
		</tr>
		<tr>
			<td colspan="10">
				<strong><xsl:value-of select="@Name" /> </strong>
			</td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
    <xsl:apply-templates select="input">
			<xsl:sort select="@InputDate"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="input">
		<tr>
			<th scope="col" colspan="10"><strong>Input</strong></th>
		</tr>
		<tr>
			<td scope="row" colspan="10">
				<strong><xsl:value-of select="@Name" /></strong>
			</td>
		</tr>
    <tr>
			<td>
				<strong>
					Container Size
				</strong>
			</td>
			<td>
				<strong>
					Serving Cost
				</strong>
			</td>
			<td>
				<strong>
					Servings Per Container
				</strong>
			</td>
			<td>
				<strong>
					Actual Servings Per Container
				</strong>
			</td>
			<td>
				<strong>
					Gender Of Serving Person
				</strong>
			</td>
			<td>
				<strong>
					Weight Of Serving Person
				</strong>
			</td>
      <td>
				<strong>
					Actual Calories Per Day
				</strong>
			</td>
			<td>
				<strong>
					Calories Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Calories From Fat Per Actual Serving
				</strong>
			</td>
      <td>
			</td>
		</tr>
		<tr>
			<td>
				<strong>
					Total Fat Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Total Fat Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Saturated Fat Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Saturated Fat Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Trans Fat Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Cholesterol Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Cholesterol Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Sodium Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Sodium Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Potassium Per Actual Serving
				</strong>
			</td>
		</tr>
		<tr>
			<td>
				<strong>
					Total Carbohydrate Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Total Carbohydrate Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Other Carbohydrate Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Other Carbohydrate Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Dietary Fiber Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Dietary Fiber Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Sugars Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Protein Per Actual Serving
				</strong>
			</td>
			<td>
				<strong>
					Protein Actual Daily Percent
				</strong>
			</td>
			<td>
				<strong>
					Vitamin A Percent Actual Daily Value
				</strong>
			</td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
		<xsl:apply-templates select="inputseries">
			<xsl:sort select="@InputDate"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="inputseries">
		<tr>
			<td scope="row" colspan="10">
				<strong>Input Series: <xsl:value-of select="@Name" /></strong>
			</td>
		</tr>
		<xsl:variable name="calculatorid"><xsl:value-of select="@CalculatorId"/></xsl:variable>
		<xsl:apply-templates select="root/linkedview[@Id=$calculatorid]">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="root/linkedview">
		<xsl:param name="localName" />
		<xsl:if test="($localName = 'inputgroup')">
      <tr>
				<td>
					<xsl:value-of select="@TContainerSize" />
				</td>
				<td>
					<xsl:value-of select="@TServingCost" />
				</td>
				<td>
					<xsl:value-of select="@TServingsPerContainer" />
				</td>
				<td>
					<xsl:value-of select="@TActualServingsPerContainer" />
				</td>
				<td>
					<xsl:value-of select="@TGenderOfServingPerson" />
				</td>
				<td>
					<xsl:value-of select="@TWeightOfServingPerson" />
				</td>
        <td>
					<xsl:value-of select="@TActualCaloriesPerDay" />
				</td>
				<td>
					<xsl:value-of select="@TCaloriesPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TCaloriesFromFatPerActualServing" />
				</td>
        <td>
				</td>
			</tr>
			<tr>
				<td>
					<xsl:value-of select="@TTotalFatPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TTotalFatActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TSaturatedFatPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TSaturatedFatActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TTransFatPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TCholesterolPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TCholesterolActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TSodiumPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TSodiumActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TPotassiumPerActualServing" />
				</td>
			</tr>
			<tr>
				<td>
					<xsl:value-of select="@TTotalCarbohydratePerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TTotalCarbohydrateActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TOtherCarbohydratePerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TOtherCarbohydrateActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TDietaryFiberPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TDietaryFiberActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TSugarsPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TProteinPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TProteinActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TVitaminAPercentActualDailyValue" />
				</td>
			</tr>
		</xsl:if>
		<xsl:if test="($localName != 'inputgroup')">
      <tr>
				<td>
					<xsl:value-of select="@ContainerSize" />
				</td>
				<td>
					<xsl:value-of select="@ServingCost" />
				</td>
				<td>
					<xsl:value-of select="@ServingsPerContainer" />
				</td>
				<td>
					<xsl:value-of select="@ActualServingsPerContainer" />
				</td>
				<td>
					<xsl:value-of select="@GenderOfServingPerson" />
				</td>
				<td>
					<xsl:value-of select="@WeightOfServingPerson" />
				</td>
        <td>
					<xsl:value-of select="@ActualCaloriesPerDay" />
				</td>
				<td>
					<xsl:value-of select="@CaloriesPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@CaloriesFromFatPerActualServing" />
				</td>
        <td>
				</td>
			</tr>
			<tr>
				<td>
					<xsl:value-of select="@TotalFatPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TotalFatActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@SaturatedFatPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@SaturatedFatActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@TransFatPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@CholesterolPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@CholesterolActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@SodiumPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@SodiumActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@PotassiumPerActualServing" />
				</td>
			</tr>
			<tr>
				<td>
					<xsl:value-of select="@TotalCarbohydratePerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@TotalCarbohydrateActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@OtherCarbohydratePerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@OtherCarbohydrateActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@DietaryFiberPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@DietaryFiberActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@SugarsPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@ProteinPerActualServing" />
				</td>
				<td>
					<xsl:value-of select="@ProteinActualDailyPercent" />
				</td>
				<td>
					<xsl:value-of select="@VitaminAPercentActualDailyValue" />
				</td>
			</tr>
      <tr>
        <td colspan="10">
					<strong>Description : </strong><xsl:value-of select="@CalculatorDescription" />
				</td>
			</tr>
		</xsl:if>
	</xsl:template>
</xsl:stylesheet>
