﻿<?xml version="1.0" encoding="UTF-8" ?>
<!-- Author: www.devtreks.org, 2014, June -->
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
	xmlns:y0="urn:DevTreks-support-schemas:Budget"
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
					<xsl:apply-templates select="budgetgroup" />
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
		<xsl:apply-templates select="budgetgroup">
			<xsl:sort select="@Name"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budgetgroup">
    <tr>
			<th scope="col" colspan="10">
				Budget Group : <xsl:value-of select="@Name" /> 
			</th>
		</tr>
    <tr>
      <th>
				Name
			</th>
      <th>
				Total
			</th>
			<th>
				Mean
			</th>
			<th>
				Median
			</th>
      <th>
				Variance
			</th>
			<th>
				Std Dev
			</th>
      <th>
			</th>
			<th>
			</th>
      <th>
			</th>
			<th>
			</th>
		</tr>
		<xsl:apply-templates select="root/linkedview[@AnalyzerType='mnstat1']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
    <xsl:apply-templates select="budget">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budget">
		<tr>
			<th scope="col" colspan="10">
				Budget : <xsl:value-of select="@Name" /> 
			</th>
		</tr>
    <tr>
      <th>
				Name
			</th>
      <th>
				Total
			</th>
			<th>
				Mean
			</th>
			<th>
				Median
			</th>
      <th>
				Variance
			</th>
			<th>
				Std Dev
			</th>
      <th>
			</th>
			<th>
			</th>
      <th>
			</th>
			<th>
			</th>
		</tr>
		<xsl:apply-templates select="root/linkedview[@AnalyzerType='mnstat1']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
    <xsl:apply-templates select="budgettimeperiod">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budgettimeperiod">
		<tr>
			<th scope="col" colspan="10">
				Time Period : <xsl:value-of select="@Name" /> 
			</th>
		</tr>
    <tr>
      <th>
				Name
			</th>
      <th>
				Total
			</th>
			<th>
				Mean
			</th>
			<th>
				Median
			</th>
      <th>
				Variance
			</th>
			<th>
				Std Dev
			</th>
      <th>
			</th>
			<th>
			</th>
      <th>
			</th>
			<th>
			</th>
		</tr>
    <xsl:apply-templates select="root/linkedview[@AnalyzerType='mnstat1']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
    <xsl:variable name="outcount" select="count(budgetoutcomes/budgetoutcome)"/>
    <xsl:if test="($outcount > 0)"> 
		  <tr>
			  <th scope="col" colspan="10">Outcomes</th>
		  </tr>
      <tr>
      <th>
				Name
			</th>
      <th>
				Total
			</th>
			<th>
				Mean
			</th>
			<th>
				Median
			</th>
      <th>
				Variance
			</th>
			<th>
				Std Dev
			</th>
      <th>
			</th>
			<th>
			</th>
      <th>
			</th>
			<th>
			</th>
		</tr>
		  <xsl:apply-templates select="budgetoutcomes" />
    </xsl:if>
    <xsl:variable name="opcount" select="count(budgetoperations/budgetoperation)"/>
    <xsl:if test="($opcount > 0)"> 
		  <tr>
			  <th scope="col" colspan="10">Operations</th>
		  </tr>
      <tr>
      <th>
				Name
			</th>
      <th>
				Total
			</th>
			<th>
				Mean
			</th>
			<th>
				Median
			</th>
      <th>
				Variance
			</th>
			<th>
				Std Dev
			</th>
      <th>
			</th>
			<th>
			</th>
      <th>
			</th>
			<th>
			</th>
		</tr>
		  <xsl:apply-templates select="budgetoperations" />
    </xsl:if>
	</xsl:template>
	<xsl:template match="budgetoutcomes">
		<xsl:apply-templates select="budgetoutcome">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budgetoutcome">
		<tr>
			<th scope="col" colspan="10"><strong>Outcome</strong></th>
		</tr>
		<tr>
			<td scope="row" colspan="10">
				<strong><xsl:value-of select="@Name" /></strong>
			</td>
		</tr>
		<xsl:apply-templates select="root/linkedview[@AnalyzerType='mnstat1']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
    <tr>
      <th>
				Name
			</th>
      <th>
				Total
			</th>
			<th>
				Mean
			</th>
			<th>
				Median
			</th>
      <th>
				Variance
			</th>
			<th>
				Std Dev
			</th>
      <th>
			</th>
			<th>
			</th>
      <th>
			</th>
			<th>
			</th>
		</tr>
    <xsl:apply-templates select="budgetoutput">
    </xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budgetoutput">
    <tr>
			<td scope="row" colspan="10">
				<strong>Output : <xsl:value-of select="@Name" /></strong>
			</td>
		</tr>
		<xsl:apply-templates select="root/linkedview[@AnalyzerType='mnstat1']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budgetoperations">
		<xsl:apply-templates select="budgetoperation">
			<xsl:sort select="@Date"/>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budgetoperation">
		<tr>
			<th scope="col" colspan="10"><strong>Operation</strong></th>
		</tr>
		<tr>
			<td scope="row" colspan="10">
				<strong><xsl:value-of select="@Name" /></strong>
			</td>
		</tr>
		<xsl:apply-templates select="root/linkedview[@AnalyzerType='mnstat1']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
    <tr>
      <th>
				Name
			</th>
      <th>
				Total
			</th>
			<th>
				Mean
			</th>
			<th>
				Median
			</th>
      <th>
				Variance
			</th>
			<th>
				Std Dev
			</th>
      <th>
			</th>
			<th>
			</th>
      <th>
			</th>
			<th>
			</th>
		</tr>
    <xsl:apply-templates select="budgetinput">
    </xsl:apply-templates>
	</xsl:template>
	<xsl:template match="budgetinput">
		<tr>
			<td scope="row" colspan="10">
				<strong>Input : <xsl:value-of select="@Name" /></strong>
			</td>
		</tr>
		<xsl:apply-templates select="root/linkedview[@AnalyzerType='mnstat1']">
			<xsl:with-param name="localName"><xsl:value-of select="local-name()" /></xsl:with-param>
		</xsl:apply-templates>
	</xsl:template>
	<xsl:template match="root/linkedview">
		<xsl:param name="localName" />
      <tr>
			  <td colspan="10">
				  Q Observations : <xsl:value-of select="@TQN"/>
			  </td>
      </tr>
			<tr>
			  <td>
				  <xsl:value-of select="@TMN1Name"/>
			  </td>
			  <td>
				  <xsl:value-of select="@TMN1Q"/>
			  </td>
			  <td>
					  <xsl:value-of select="@TMN1Mean"/>
			  </td>
			  <td>
					  <xsl:value-of select="@TMN1Median"/>
			  </td>
			  <td>
					  <xsl:value-of select="@TMN1Variance"/>
			  </td>
        <td>
					  <xsl:value-of select="@TMN1StandDev"/>
			  </td>
        <td colspan="4">
			  </td>
		  </tr>
      <tr>
			  <td>
				  <xsl:value-of select="@TMN2Name"/>
			  </td>
			  <td>
				  <xsl:value-of select="@TMN2Q"/>
			  </td>
			  <td>
					  <xsl:value-of select="@TMN2Mean"/>
			  </td>
			  <td>
					  <xsl:value-of select="@TMN2Median"/>
			  </td>
			  <td>
					  <xsl:value-of select="@TMN2Variance"/>
			  </td>
        <td>
					  <xsl:value-of select="@TMN2StandDev"/>
			  </td>
        <td colspan="4">
			  </td>
		  </tr>
      <xsl:if test="(string-length(@TMN3Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN3Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN3Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN3Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN3Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN3Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN3StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
      <xsl:if test="(string-length(@TMN4Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN4Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN4Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN4Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN4Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN4Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN4StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
      <xsl:if test="(string-length(@TMN5Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN5Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN5Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN5Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN5Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN5Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN5StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
      <xsl:if test="(string-length(@TMN6Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN6Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN6Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN6Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN6Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN6Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN6StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
      <xsl:if test="(string-length(@TMN7Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN7Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN7Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN7Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN7Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN7Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN7StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
      <xsl:if test="(string-length(@TMN8Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN8Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN8Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN8Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN8Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN8Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN8StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
      <xsl:if test="(string-length(@TMN9Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN9Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN9Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN9Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN9Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN9Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN9StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
      <xsl:if test="(string-length(@TMN10Name) > 0)">
        <tr>
			    <td>
				    <xsl:value-of select="@TMN10Name"/>
			    </td>
			    <td>
				    <xsl:value-of select="@TMN10Q"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN10Mean"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN10Median"/>
			    </td>
			    <td>
					    <xsl:value-of select="@TMN10Variance"/>
			    </td>
          <td>
					    <xsl:value-of select="@TMN10StandDev"/>
			    </td>
          <td colspan="4">
			    </td>
		    </tr>
      </xsl:if>
	</xsl:template>
</xsl:stylesheet>
